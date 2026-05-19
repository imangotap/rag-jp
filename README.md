# 🇯🇵 Japanese RAG System
日本語対応の RAG（Retrieval-Augmented Generation）システムです。  
Hybrid Search + RRF + Reranking を組み合わせた高精度な検索・回答生成を実現します。
---
## アーキテクチャ
```
[ファイルアップロード]
        ↓
   テキスト抽出 (PDF / TXT)
        ↓
   チャンク分割 (500文字 / overlap 50)
        ↓
   ベクトル化 (BGE-M3) → ChromaDB に保存
   トークン化 (Janome)  → BM25 インデックスに保存
[クエリ]
        ↓
   ┌─────────────────────────────┐
   │  Dense Search (BGE-M3)      │  意味的な類似度
   │  Sparse Search (BM25+Janome)│  キーワードマッチ
   └─────────────────────────────┘
        ↓
   RRF（Reciprocal Rank Fusion）でスコア融合
        ↓
   Reranker (bge-reranker-v2-m3) で再順位付け
        ↓
   Context Window Expansion（前後チャンクを取得）
        ↓
   LLM (Ollama / qwen2.5:7b) で回答生成
```
---
## 技術スタック
| カテゴリ | 技術 |
|----------|------|
| Embedding | BAAI/BGE-M3 |
| Vector DB | ChromaDB |
| Sparse Search | BM25 (rank-bm25) |
| 日本語トークナイザ | Janome |
| Reranker | BAAI/bge-reranker-v2-m3 |
| LLM | Ollama (qwen2.5:7b) |
| API Framework | FastAPI |
---
## 工夫した点
### 1. 日本語特化のトークナイザ
Janome を使って品詞フィルタリングを実施。  
名詞・動詞・形容詞のみを抽出することで、BM25 の検索精度を向上。
```python
if part_of_speech in ["名詞", "動詞", "形容詞"]:
    words.append(token.surface)
```
### 2. Hybrid Search + RRF
Dense（意味）と Sparse（キーワード）の両方の強みを活かし、  
RRF（k=60）でスコアを融合することでバランスの良い検索を実現。
### 3. 段階的なデバッグ情報
各検索ステージの中間結果を返すことで、精度が悪い場合の原因特定を容易にしました。
```json
{
  "dense_search_results":  [...],
  "sparse_search_results": [...],
  "rrf_search_results":    [...],
  "final_top_ids":         [...]
}
```
### 4. Context Window Expansion
上位チャンクの前後チャンクも取得することで、文脈の途切れを防ぎます。
---
## APIエンドポイント

| メソッド | パス | 説明 |
|----------|------|------|
| POST | `/upload` | PDF / TXT ファイルをアップロード・インデックス化 |
| GET | `/hybrid` | Hybrid Search でドキュメントを検索 |
| GET | `/retrieve` | Dense Search のみで検索 |
| POST | `/chat` | 会話履歴付きRAG チャット |
APIドキュメント（Swagger UI）：`http://localhost:8000/docs`
---
## セットアップ
### 1. 依存ライブラリのインストール
```bash
pip install -r requirements.txt
```
### 2. Ollama のセットアップ
```bash
# Ollama をインストール後
ollama pull qwen2.5:7b
```
### 3. サーバー起動
```bash
uvicorn main:app --reload
```
### 4. ブラウザで確認
```
http://localhost:8000/docs
```
---
## 今後の改善予定
- [ ] RAGAS による定量評価の追加
- [ ] Azure OpenAI / OpenAI API 対応
- [ ] チャンクサイズの自動最適化
- [ ] フロントエンド UI の追加
