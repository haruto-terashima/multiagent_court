from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data_after_parce" / "chunked"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data_after_parce" / "index"
DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-base"
BM25_K1 = 1.5
BM25_B = 0.75
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[一-龯ぁ-んァ-ヶー]+")


def load_jsonl_records(data_dir: Path, limit: int | None = None) -> Iterable[dict]:
    count = 0

    if not data_dir.exists():
        raise FileNotFoundError(
            f"{data_dir} does not exist. Run preprocess first, for example: "
            "python3 preprocess.py --cases data/hanrei_data --laws data/hourei_data"
        )

    for path in sorted(data_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc

                text = (record.get("text") or "").strip()
                if not text:
                    continue

                record["text"] = text
                yield record
                count += 1

                if limit is not None and count >= limit:
                    return


def passage_text(text: str) -> str:
    return f"passage: {text}"


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []

    for match in TOKEN_RE.finditer(text.lower()):
        term = match.group(0)
        if re.fullmatch(r"[A-Za-z0-9_]+", term):
            tokens.append(term)
            continue

        if len(term) == 1:
            tokens.append(term)
            continue

        if len(term) <= 12:
            tokens.append(term)

        for n in range(2, 5):
            if len(term) >= n:
                tokens.extend(term[i : i + n] for i in range(len(term) - n + 1))

    return tokens


def add_bm25_document(
    *,
    postings: dict[str, list[tuple[int, int]]],
    doc_lengths: list[int],
    source_types: list[str],
    doc_id: int,
    record: dict,
) -> None:
    tokens = tokenize(record.get("text", ""))
    counts = Counter(tokens)

    doc_lengths.append(len(tokens))
    source_types.append(record.get("source_type", "document"))

    for term, tf in counts.items():
        postings[term].append((doc_id, tf))


def finalize_bm25(
    *,
    postings: dict[str, list[tuple[int, int]]],
    doc_lengths: list[int],
    source_types: list[str],
) -> dict:
    doc_count = len(doc_lengths)
    avgdl = sum(doc_lengths) / doc_count if doc_count else 0.0
    idf = {
        term: math.log(1 + (doc_count - len(term_postings) + 0.5) / (len(term_postings) + 0.5))
        for term, term_postings in postings.items()
    }

    return {
        "postings": dict(postings),
        "doc_lengths": doc_lengths,
        "source_types": source_types,
        "avgdl": avgdl,
        "doc_count": doc_count,
        "idf": idf,
        "k1": BM25_K1,
        "b": BM25_B,
        "tokenizer": "regex_word_plus_ja_bigrams",
    }


def encode_batch(
    model,
    texts: list[str],
    batch_size: int,
) -> Any:
    import numpy as np

    embeddings = model.encode(
        [passage_text(text) for text in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype="float32")


def write_pickle(path: Path, value) -> None:
    with path.open("wb") as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)


def flush_batch(
    *,
    model,
    index,
    texts: list[str],
    batch_size: int,
) -> Any:
    import faiss

    embeddings = encode_batch(model, texts, batch_size=batch_size)

    if index is None:
        index = faiss.IndexFlatIP(embeddings.shape[1])

    index.add(embeddings)
    return index


def build_index(
    *,
    data_dir: Path,
    out_dir: Path,
    model_name: str,
    batch_size: int,
    encode_batch_size: int,
    limit: int | None = None,
) -> int:
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        from tqdm import tqdm
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Missing dependency: {exc.name}. Install project dependencies with "
            "python3 -m pip install -r requirements.txt"
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(model_name)
    index: Any | None = None
    texts: list[str] = []
    meta: list[dict] = []
    pending_texts: list[str] = []
    bm25_postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    bm25_doc_lengths: list[int] = []
    bm25_source_types: list[str] = []

    records = load_jsonl_records(data_dir, limit=limit)

    for record in tqdm(records, desc="indexing chunks", unit="chunk"):
        text = record["text"]
        doc_id = len(texts)
        texts.append(text)
        meta.append(record)
        pending_texts.append(text)
        add_bm25_document(
            postings=bm25_postings,
            doc_lengths=bm25_doc_lengths,
            source_types=bm25_source_types,
            doc_id=doc_id,
            record=record,
        )

        if len(pending_texts) >= batch_size:
            index = flush_batch(
                model=model,
                index=index,
                texts=pending_texts,
                batch_size=encode_batch_size,
            )
            pending_texts = []

    if pending_texts:
        index = flush_batch(
            model=model,
            index=index,
            texts=pending_texts,
            batch_size=encode_batch_size,
        )

    if index is None or not texts:
        raise ValueError(f"No chunks found in {data_dir}.")

    faiss.write_index(index, str(out_dir / "faiss.index"))
    write_pickle(out_dir / "meta.pkl", meta)
    write_pickle(out_dir / "texts.pkl", texts)
    write_pickle(
        out_dir / "bm25.pkl",
        finalize_bm25(
            postings=bm25_postings,
            doc_lengths=bm25_doc_lengths,
            source_types=bm25_source_types,
        ),
    )

    manifest = {
        "model_name": model_name,
        "embedding_prefix": "passage: ",
        "normalize_embeddings": True,
        "faiss_index": "IndexFlatIP",
        "chunk_count": len(texts),
        "data_dir": str(data_dir),
        "bm25": {
            "path": "bm25.pkl",
            "k1": BM25_K1,
            "b": BM25_B,
            "tokenizer": "regex_word_plus_ja_bigrams",
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"indexed {len(texts)} chunks into {out_dir / 'faiss.index'}")
    return len(texts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FAISS index from chunked JSONL files.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=512, help="Chunks to add to FAISS at once.")
    parser.add_argument("--encode-batch-size", type=int, default=32, help="SentenceTransformer encode batch size.")
    parser.add_argument("--limit", type=int, help="Index only the first N chunks for testing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        build_index(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            model_name=args.model_name,
            batch_size=args.batch_size,
            encode_batch_size=args.encode_batch_size,
            limit=args.limit,
        )
    except (FileNotFoundError, ModuleNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
