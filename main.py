"""
システム構成：
fastapi -> Uploadfile , Chunker , VectorStore , Retrieve , chat , history
FastAPI → ファイルアップロード → テキスト分割 → ベクトルストア保存 → 検索 → 再ランク付け → 会話生成 → 履歴管理

ハイブリッド検索の仕組み：
１．密ベクトル検索(Dense): BGEーM3　埋め込みモデルを使用、意味的な類似度で検索
２.疎ベクトル検索(sparse/bm25) :janomeによる日本語トークナイズ、キーワード一致度で検索
３.RRF融合 :　両者の結果を「順位」ベースで統合、最適な結果を抽出

デバッグ機能：
精度改善のため、各段階(Dense/Sparse/RRF)スコアと順位を記録
問題発生時にどの処理で精度が低下しているか特定可能
"""
from fastapi import FastAPI,UploadFile,File
from fastapi.middleware.cors import CORSMiddleware
import io

import chromadb
from sentence_transformers import SentenceTransformer
import ollama
from rank_bm25 import BM25Okapi
import json
from FlagEmbedding import FlagReranker

#  ===========  日本語トークナイザー設定　===============
from janome.tokenizer import Tokenizer

tokenizer = Tokenizer()
def japanese_tokenizer(text:str) -> list[str]:
    if not text:
        return []
    # 文章をトークン（単語の断片）に分割する
    tokens = tokenizer.tokenize(text)
    words= []
    for token in tokens:
        # 品詞情報を取得
        part_of_speech = token.part_of_speech.split(",")[0]
        #　検索に重要な品詞だけを選別
        if part_of_speech in ["名詞","動詞","形容詞"]:
            words.append(token.surface)
    return words
# ================== chroma,embedding_model ========================
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="knowledge")
embedding_model = SentenceTransformer("BAAI/BGE-M3")

# FastAPIアプリケーション本体を作成
app = FastAPI()
# CORS設定:外部からのAPIアクセスを許可
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ================= BM２５インデックス管理　===============
bm25_corpus = []
bm25_index  = None
def rebuild_bm25():
    """
    BM２５のインデックスを再構築する関数
    文章を追加後など、データ更新時に実行して検索可能にする
    """
    global bm25_index
    if not bm25_corpus:
        bm25_index = None
        return
    #　　各文書を日本語トークナイザーで単語文割してBM2５のコーパスに変換
    tokenized = [japanese_tokenizer(doc["text"]) for doc in bm25_corpus]
    bm25_index = BM25Okapi(tokenized)
# ================== hybrid search ================
@app.get("/hybrid",)
def hybrid_search(query:str,top_k:int=10):
    """
    ハイブリッド検索のメイン処理
    １.密ベクトル検索　2.BM２５キーワード検索　3.RRF融合　-> return
    """
    #==========  Dense search
    # クエリをベクトルに変換
    query_embedding =  embedding_model.encode(query).tolist()
    # ChromaDBでベクトル検索、必要数の２倍取得して融合の候補を確保
    dense_results = collection.query(
        query_embeddings = [query_embedding],
        n_results = top_k * 2,
    )
    dense_scores = {}
    dense_debug = []
    for i,doc_id in enumerate(dense_results["ids"][0]):
        #　ベクトル間の距離を取得(０に近いほど類似度が高い)
        distance = dense_results["distances"][0][i]
        #　距離をスコアに変換
        score = 1 / ( 1 + distance)
        dense_scores[doc_id] = score
        #　デバック情報に記録
        dense_debug.append({
            "doc_id":doc_id,
            "distance":round(distance,4),
            "dense_score":round(score,4),
            "dense_rank":i+1
        })
    #============ BM25  sparse search
    sparse_scores = {}
    sparse_debug = []
    if bm25_index:
        #　クエリを単語文割
        tokenized_query = japanese_tokenizer(query)
        #　全文書に対するスコア計算
        raw_scores = bm25_index.get_scores(tokenized_query)
        #　スコアの高い順に並ぶ
        top_k_indices = sorted(range(len(raw_scores)),key=lambda x:raw_scores[x],reverse=True)[:top_k]
        #　順位つけしながら結果を格納
        for rank,i in enumerate (top_k_indices):
            doc_id = bm25_corpus[i]["id"]
            score = raw_scores[i]
            sparse_scores[doc_id] = float(score)
            sparse_debug.append({
                "doc_id":doc_id,
                "raw_score":round(score,4),
                "sparse_rank":rank+1,
            })

    #============ RRF 融合
    #　全IDを集約 (重複なし)
    all_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
    #　スコア順位を並ぶ
    dense_ranked  = sorted(dense_scores ,key=lambda x: dense_scores[x], reverse=True)
    sparse_ranked = sorted(sparse_scores,key=lambda x: sparse_scores[x],reverse=True)

    rrf_scores = {}
    rrf_debug  = []
    for doc_id in all_ids:
        #　各検索での順位を取得、dense/sparse_rankedに存在しないの場合は後回し
        d_rank = dense_ranked.index(doc_id) +1 if doc_id in dense_ranked else 999
        s_rank = sparse_ranked.index(doc_id)+1 if doc_id in sparse_ranked else 999
        #　RRF計算式：１/(順位　＋　定数)　＋　1 / (順位　＋定数)
        #　定数６０：順位の影響度を調整、値が大きいほど順位差の影響が緩和される
        #　最大値はやく 0.0327　　この値に近いほどスコアが高い
        rrf_score = 1 / (d_rank + 60) + 1 / (s_rank + 60)
        rrf_scores[doc_id] = rrf_score
        rrf_debug.append({
            "doc_id":doc_id,
            "d_rank":d_rank,
            "s_rank":s_rank,
            "rrf_score":round(rrf_score,4),
        })
    #　融合スコアの高い順に上位top_k件を抽出
    top_ids = sorted(rrf_scores,key=lambda x: rrf_scores[x],reverse=True)[:top_k]
    #　IDから文書データを引くため辞書を作成
    id_to_doc = {doc["id"]:doc for doc in bm25_corpus}
    #結果リストを作成
    final_result = [
        {
            "doc_id":doc_id,
            "text":id_to_doc[doc_id]["text"],
            "metadata":id_to_doc[doc_id]["metadata"],
            "rrf_score":round(rrf_scores[doc_id],4)

        }   for doc_id in top_ids if doc_id in id_to_doc
    ]
    #　デバック情報
    debug_info = {
        "query":query,
        "dense_search_results":sorted(dense_debug,key=lambda x:x["dense_rank"],reverse=True),
        "sparse_search_results":sorted(sparse_debug,key=lambda x:x["sparse_rank"],reverse=True),
        "rrf_search_results":sorted(rrf_debug,key=lambda x:x["rrf_score"],reverse=True),
        "final_top_ids":top_ids
    }
    return {
        "final_result":final_result,
        "debug_info":debug_info
    }
#============== expand_context ================
#コンテキスト展開処理
def expand_context(chunk_id:str,window:int = 1) ->str:

    # filename_chunk_5 -> filename,5
    parts = chunk_id.rsplit("_chunk_",1)
    filename = parts[0]
    index    = int(parts[1])

    texts = []
    # 5-1,5+1+1 (4,7) ->456
    for i in range(index-window,index+window+1):
        neighbor_id = f"{filename}_chunk_{i}"
        try: # データベースからIDを取得
            result = collection.get(ids=[neighbor_id])
            if result["documents"]:
                texts.append(result["documents"][0])
        except Exception:
            #　先頭チャンクの前、最後チャンクの後など存在しないの場合はスキップ
            continue
    return "\n".join(texts)
# =================== Rerank ====================
# 再ランク付け処理
reranker = FlagReranker("BAAI/bge-reranker-v2-m3",use_fp16=True)

def rerank(query:str,chunks:list,top_n:int=3) ->list:
    """
    検索結果をクエリとの関連度で再評価し、上位ｎ件を抽出
    ハイブリッド検索の結果を更に絞り込む、回答精度を向上
    """
    #　クエリと各文書のペアを作成
    pairs = [[query,chunk["text"]] for chunk in chunks]
    #　各ペアの関連度スコアを計算
    score = reranker.compute_score(pairs)
    #　各チャンクにスコアを追加
    for i,chunk in enumerate(chunks):
        chunk["rerank_score"] = round(float(score[i]),4)
    #上位top_n件を返却
    return sorted(chunks,key=lambda x:x["rerank_score"],reverse=True)[:top_n]

# ================== upload =====================
# upload file (async)
@app.post("/upload")
async def upload(file:UploadFile=File(...)):
    """
    ファイルを受け取り、デキスト抽出、文割、ベクトル化、chromaDB/bm25DB保存
    """
    content = await file.read()
    full_text  = extract_text(file.filename,content)
    chunks = chunk_text(full_text)

    # str to vector
    texts = [chunk["text"]for chunk in chunks]
    embeddings = embedding_model.encode(texts).tolist()

    # save to ChromaDB
    collection.add(
        ids = [f"{file.filename}_chunk_{chunk['index']}" for chunk in chunks],
        documents = [chunk['text'] for chunk in chunks],
        embeddings = embeddings,
        metadatas =[{"filename":file.filename ,"index":chunk['index']} for chunk in chunks],
    )
    # save to BM25
    for chunk in chunks:
        bm25_corpus.append({
            "id"   : f"{file.filename}_chunk_{chunk['index']}",
            "text" : chunk["text"],
            "metadata":{"filename":file.filename,"index":chunk["index"]}
        })
    rebuild_bm25()  # データ追加後にBM２５インデックスを再構築して検索可能にする

    return {
        "filename":file.filename,
        "text_length":len(full_text),
        "chunk_count":len(chunks),
        "chunks_preview":chunks[:3],
        "status":"Saved Chroma_DB"
    }
#=====================  extract_text ========================
def extract_text(filename,content):
    if filename.endswith(".txt"):
        return content.decode("utf-8",errors="ignore")

    elif filename.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text)
        return "\n".join(pages)
    else:
        raise ValueError(f"対応対象外タイプ{filename}")

#=========================  chunk_text ==========================
def chunk_text(text:str,chunk_size:int=500,overlap:int=50)->list[dict]:
    """
    長いテキストを検索に適した大きさのチャンクに文割する
    - chunk_size: チャンクの最大文字数
    - overlap :  次のチャンクとの重複文字数（文脈の途切れを防ぐ）

    """
    chunks = []
    start  = 0
    index  = 0

    while start < len(text):
        #　終了位置は　スタット+サイズ　または　テキスト末尾のいずれ小さい方
        end = min(start + chunk_size,len(text))
        chunk = text[start:end]
        #　空白だけのチャンクはスキップ
        if chunk.strip():
            chunks.append({
                "index":index,
                "text":chunk,
                "start":start,
                "end":end,
            })
            index  += 1
        #　次の位置は：サイズー重複分だけ進める
        start += chunk_size - overlap
    return chunks

# ========================== retrieve ===========================
@app.get("/retrieve")
def retrieve(query:str,top_k:int=10):

    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings = [query_embedding],
        n_results = top_k
    )
    retrieve_chunks = []
    if results["metadatas"]:
        for i in range(len(results["documents"][0])):
             retrieve_chunks.append({
                 "text":results["documents"][0][i],
                 "metadata":results["metadatas"][0][i],
                 "distance":results["distances"][0][i],
            })
    return {
        "query":query,
        "results": retrieve_chunks,
    }

# ========================= chat ======================
#会話処理
from pydantic import BaseModel
from typing import List
#　リクエストのデータ型定義：
class Message(BaseModel):
    role: str     # system / user /assitant
    content: str  # message
#　会話リクエスト全体のデータ型定義
class ChatRequest(BaseModel):
    query: str                    # 今回のユーザーの質問
    history: List[Message] = []   # 会話の歴史
@app.post("/chat")
def chat(request:ChatRequest):
    # query_embedding = embedding_model.encode(request.query).tolist()
    # results = collection.query(
    #     query_embeddings = [query_embedding],
    #     n_results = 3,
    # )
    # if not  results.get("metadatas") or not results["documents"][0]:
    #     return {
    #         "query" : request.query,
    #         "answer":"関連する資料が見つかりませんでした。",
    #         "sources": [],
    #         "history": request.history,
    #     }
    # context = "\n\n".join(results["documents"][0])
    # hybrid 検索で関連文を取得
    result = hybrid_search(query=request.query,top_k=10)
    chunks = result["final_result"]
    #　リランク付けで更に関連度高い上位３件を絞り込む　
    reranked = rerank(request.query,chunks,top_n=3)
    #  文脈展開：前後チャンクを結合して長い文脈を作成
    context = "\n\n".join([
    expand_context(chunk["metadata"]["filename"]+ "_chunk_" + str(chunk["metadata"]["index"]))
    for chunk in reranked
    ])
    #context = "\n\n".join([chunk['text'] for chunk in chunks[:3]])

    # プロンプト作成　：　LLMへの指示文
    system_prompt = (
        "あなたは知識庫アシスタントです。以下の【知識庫】に基づいて回答してください。"
        "知識庫にない内容は「資料にないため回答できません」と述べてください。"
        "勝手に作り話は絶対にしないでください。")
    user_prompt = f"""
    【知識庫】
    {context}
    【質問】
    {request.query}
    """
    messages = [{"role":"system","content":system_prompt}]
    # history
    messages.extend([{"role":msg.role,"content":msg.content} for msg in  request.history])
    # curren
    messages.append({"role":"user","content":user_prompt})
    # LLM
    response = ollama.chat(
        model = "qwen2.5:7b",
        messages = messages,
        options = {"temperature":0.0}
    )
    # 回答本文を抽出
    answer_text = response["message"]["content"]

    updated_history = [
        *[{"role":msg.role,"content":msg.content} for msg in request.history],
        {"role":"user","content":request.query},
        {"role":"assistant","content":answer_text},
    ]

    return {
        "query"  :request.query,
        "answer" :answer_text,
        "sources":[chunk["metadata"] for chunk in reranked],
        "history":updated_history,
    }

def rebuild_bm25_from_chroma():
    """
    サーバー起動時にchromaに保存されているデータを読み込み、
    BM２５のインデックスを再構築する
    サーバー再起動後もハイブリッド検索が使えるようにする
    """
    global bm25_corpus
    bm25_corpus = [] # clear
    # Chromaから全部取得
    results = collection.get()
    if not results["documents"]:
        return

    for i, doc in enumerate(results["documents"]):
        bm25_corpus.append({
            "id"      : results["ids"][i],
            "text"    : doc,
            "metadata": results["metadatas"][i],
        })
    rebuild_bm25()
# モジュール読み込み時に自動実行
rebuild_bm25_from_chroma()
