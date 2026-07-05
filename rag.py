import faiss
import pickle
import numpy as np
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[一-龯ぁ-んァ-ヶー]+")


def tokenize(text: str) -> list[str]:
    tokens = []

    for match in TOKEN_RE.finditer((text or "").lower()):
        term = match.group(0)
        if re.fullmatch(r"[A-Za-z0-9_]+", term):
            tokens.append(term)
        elif len(term) == 1:
            tokens.append(term)
        else:
            if len(term) <= 12:
                tokens.append(term)

            for n in range(2, 5):
                if len(term) >= n:
                    tokens.extend(term[i : i + n] for i in range(len(term) - n + 1))

    return tokens


class RAGEngine:
    def __init__(
        self,
        index_path="data_after_parce/index/faiss.index",
        meta_path="data_after_parce/index/meta.pkl",
        texts_path="data_after_parce/index/texts.pkl",
        bm25_path="data_after_parce/index/bm25.pkl",
        model_name="intfloat/multilingual-e5-base",
    ):
        for path in [index_path, meta_path, texts_path]:
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"RAG artifact not found: {path}. Run python3 preprocess.py and python3 emb_db/build_index.py first."
                )

        self.index = faiss.read_index(str(index_path))

        with open(meta_path, "rb") as f:
            self.meta = pickle.load(f)

        with open(texts_path, "rb") as f:
            self.texts = pickle.load(f)

        self.bm25 = None
        if Path(bm25_path).exists():
            with open(bm25_path, "rb") as f:
                self.bm25 = pickle.load(f)

        self.model = SentenceTransformer(model_name)

    def retrieve(self, query, k=5, source_type=None, mode="hybrid", fetch_k=None):
        fetch_k = fetch_k or max(k * 20, 50)

        if mode == "embedding" or not self.bm25:
            return self.retrieve_embedding(query=query, k=k, source_type=source_type, fetch_k=fetch_k)

        embedding_hits = self.retrieve_embedding_candidates(
            query=query,
            source_type=source_type,
            fetch_k=fetch_k,
        )
        bm25_hits = self.retrieve_bm25_candidates(
            query=query,
            source_type=source_type,
            fetch_k=fetch_k,
        )

        fused = self.fuse_candidates(embedding_hits, bm25_hits)
        return [self.result_from_doc_id(doc_id, score) for doc_id, score in fused[:k]]

    def retrieve_embedding(self, query, k=5, source_type=None, fetch_k=None):
        candidates = self.retrieve_embedding_candidates(
            query=query,
            source_type=source_type,
            fetch_k=fetch_k or max(k * 20, 50),
        )
        return [
            self.result_from_doc_id(doc_id, score, embedding_score=score)
            for doc_id, score in candidates[:k]
        ]

    def retrieve_embedding_candidates(self, query, source_type=None, fetch_k=50):
        q = self.model.encode([f"query: {query}"], normalize_embeddings=True)
        q = np.asarray(q, dtype="float32")

        scores, ids = self.index.search(q, min(fetch_k, len(self.texts)))

        candidates = []
        for score, i in zip(scores[0], ids[0]):
            if i < 0:
                continue

            meta = self.meta[i]
            if source_type and meta.get("source_type") != source_type:
                continue

            candidates.append((int(i), float(score)))

        return candidates

    def retrieve_bm25_candidates(self, query, source_type=None, fetch_k=50):
        if not self.bm25:
            return []

        postings = self.bm25["postings"]
        doc_lengths = self.bm25["doc_lengths"]
        source_types = self.bm25.get("source_types", [])
        idf = self.bm25["idf"]
        avgdl = self.bm25["avgdl"] or 1.0
        k1 = self.bm25.get("k1", 1.5)
        b = self.bm25.get("b", 0.75)

        scores = {}
        for term in set(tokenize(query)):
            term_postings = postings.get(term)
            if not term_postings:
                continue

            term_idf = idf.get(term, 0.0)
            for doc_id, tf in term_postings:
                if source_type and source_types and source_types[doc_id] != source_type:
                    continue

                dl = doc_lengths[doc_id] or 1
                denom = tf + k1 * (1 - b + b * dl / avgdl)
                scores[doc_id] = scores.get(doc_id, 0.0) + term_idf * (tf * (k1 + 1) / denom)

        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:fetch_k]

    def fuse_candidates(self, embedding_hits, bm25_hits, rrf_k=60):
        fused = {}

        for rank, (doc_id, _score) in enumerate(embedding_hits, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)

        for rank, (doc_id, _score) in enumerate(bm25_hits, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)

        return sorted(fused.items(), key=lambda item: item[1], reverse=True)

    def result_from_doc_id(self, doc_id, score, embedding_score=None, bm25_score=None):
        result = {
            "text": self.texts[doc_id],
            "meta": self.meta[doc_id],
            "score": float(score),
        }
        if embedding_score is not None:
            result["embedding_score"] = float(embedding_score)
        if bm25_score is not None:
            result["bm25_score"] = float(bm25_score)
        return result

    def retrieve_text(self, query, k=5, source_type=None, mode="hybrid"):
        results = self.retrieve(query=query, k=k, source_type=source_type, mode=mode)
        return format_results(results)


def format_results(results):
    blocks = []

    for idx, result in enumerate(results, start=1):
        meta = result["meta"]
        title = (
            meta.get("title")
            or meta.get("law_name")
            or meta.get("case_id")
            or meta.get("source_path")
            or "unknown"
        )
        section = meta.get("section") or meta.get("article_title") or meta.get("article_num") or ""
        source_type = meta.get("source_type", "document")
        score = result.get("score", 0.0)

        blocks.append(
            f"[{idx}] type={source_type} score={score:.4f} title={title} section={section}\n"
            f"{result['text']}"
        )

    return "\n\n".join(blocks)
