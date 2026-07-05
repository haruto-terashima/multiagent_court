# LLM Court RAG

民事裁判シミュレーションに、判例HTMLと法令XMLを使ったRAG検索を組み込むプロジェクトです。

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
export GEMINI_API_KEY="your-api-key"
```

Ubuntu/Debianで `venv` が使えない場合は、先に `python3-venv` をインストールしてください。

## Build RAG Data

判例HTMLと法令XMLをチャンク化します。

```bash
python3 preprocess.py --cases data/hanrei_data --laws data/hourei_data
```

データセットを別の場所に置いている場合は、そのパスを指定してください。入力パスが存在しない場合は警告を出して 0 件のJSONLを生成します。

```bash
python3 preprocess.py \
  --cases /path/to/hanrei_html \
  --laws /path/to/hourei_xml \
  --out-dir data_after_parce/chunked
```

出力JSONLには、本文 `text` と、後続の検索・引用で使う `source_type`、`source_path`、条文番号、判例セクション、`chunk_index` などのメタデータが入ります。

FAISSインデックスを作成します。

```bash
python3 emb_db/build_index.py
```

生成物:

```text
data_after_parce/chunked/cases.jsonl
data_after_parce/chunked/laws.jsonl
data_after_parce/index/faiss.index
data_after_parce/index/meta.pkl
data_after_parce/index/texts.pkl
data_after_parce/index/bm25.pkl
data_after_parce/index/manifest.json
```

## Run

```bash
python3 main.py
```

`main.py` の `case` を変更すると、別の事案で裁判フローを実行できます。
