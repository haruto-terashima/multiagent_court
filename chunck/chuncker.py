from __future__ import annotations

import re
from typing import Iterable


LAW_MAX_CHARS = 900
CASE_MAX_CHARS = 1200
OVERLAP_CHARS = 160
MIN_CHUNK_CHARS = 120

SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[。．！？!?])")


def normalize_text(text: str | None) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in (text or "").splitlines()]
    return "\n".join(line for line in lines if line)


def split_long_unit(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            candidates = [
                text.rfind("。", start, end),
                text.rfind("、", start, end),
                text.rfind("，", start, end),
                text.rfind(" ", start, end),
            ]
            split_at = max(candidates)
            if split_at > start + max_chars // 2:
                end = split_at + 1

        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        start = end

    return pieces


def split_into_units(text: str, max_chars: int) -> list[str]:
    units: list[str] = []

    for line in normalize_text(text).splitlines():
        if len(line) <= max_chars:
            units.append(line)
            continue

        sentence_parts = [part.strip() for part in SENTENCE_BOUNDARY_RE.split(line) if part.strip()]
        if len(sentence_parts) <= 1:
            units.extend(split_long_unit(line, max_chars))
            continue

        for sentence in sentence_parts:
            units.extend(split_long_unit(sentence, max_chars))

    return units


def tail_overlap(text: str, overlap_chars: int) -> str:
    if overlap_chars <= 0 or len(text) <= overlap_chars:
        return text

    tail = text[-overlap_chars:]
    newline = tail.find("\n")
    if newline >= 0 and newline + 1 < len(tail):
        return tail[newline + 1 :].strip()
    return tail.strip()


def chunk_text(
    text: str,
    *,
    max_chars: int,
    overlap_chars: int = OVERLAP_CHARS,
    min_chunk_chars: int = MIN_CHUNK_CHARS,
) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for unit in split_into_units(text, max_chars):
        unit_len = len(unit)
        separator_len = 1 if current else 0

        if current and current_len + separator_len + unit_len > max_chars:
            chunk = "\n".join(current).strip()
            if chunk:
                chunks.append(chunk)

            overlap = tail_overlap(chunk, overlap_chars)
            if overlap and overlap != unit and len(overlap) + 1 + unit_len <= max_chars:
                current = [overlap, unit]
                current_len = len(overlap) + 1 + unit_len
            else:
                current = [unit]
                current_len = unit_len
            continue

        current.append(unit)
        current_len += separator_len + unit_len

    if current:
        chunks.append("\n".join(current).strip())

    return merge_tiny_tail(chunks, max_chars=max_chars, min_chunk_chars=min_chunk_chars)


def merge_tiny_tail(chunks: list[str], *, max_chars: int, min_chunk_chars: int) -> list[str]:
    if len(chunks) < 2 or len(chunks[-1]) >= min_chunk_chars:
        return chunks

    previous = chunks[-2]
    tail = chunks[-1]
    merged = f"{previous}\n{tail}".strip()

    if len(merged) <= max_chars:
        return [*chunks[:-2], merged]

    return chunks


def make_chunk_record(
    record: dict,
    text: str,
    index: int,
    total: int,
    strategy: str,
    char_start: int | None,
    char_end: int | None,
) -> dict:
    chunk = record.copy()
    chunk["text"] = text
    chunk["chunk_index"] = index
    chunk["chunk_count"] = total
    chunk["chunk_strategy"] = strategy
    chunk["chunk_char_count"] = len(text)
    chunk["chunk_char_start"] = char_start
    chunk["chunk_char_end"] = char_end
    return chunk


def build_chunk_records(record: dict, chunks: Iterable[str], strategy: str) -> list[dict]:
    clean_chunks = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
    total = len(clean_chunks)
    original_text = normalize_text(record.get("text"))
    cursor = 0
    records: list[dict] = []

    for index, chunk in enumerate(clean_chunks):
        start = original_text.find(chunk, max(0, cursor - OVERLAP_CHARS))
        if start < 0:
            start = original_text.find(chunk)

        if start >= 0:
            end = start + len(chunk)
            cursor = end
            char_start: int | None = start
            char_end: int | None = end
        else:
            char_start = None
            char_end = None

        records.append(make_chunk_record(record, chunk, index, total, strategy, char_start, char_end))

    return records


def chunk_case_record(record: dict) -> list[dict]:
    text = normalize_text(record.get("text"))
    section = record.get("section", "")

    if not text:
        return []

    if "主文" in section:
        return build_chunk_records({**record, "text": text}, [text], "case_preserve_disposition")

    chunks = chunk_text(
        text,
        max_chars=CASE_MAX_CHARS,
        overlap_chars=OVERLAP_CHARS,
        min_chunk_chars=MIN_CHUNK_CHARS,
    )
    return build_chunk_records({**record, "text": text}, chunks, "case_sentence_overlap")


def chunk_law_record(record: dict) -> list[dict]:
    text = normalize_text(record.get("text"))

    if not text:
        return []

    text_part = record.get("text_part", "")
    if len(text) <= LAW_MAX_CHARS:
        strategy = "law_structural_unit"
        return build_chunk_records({**record, "text": text}, [text], strategy)

    overlap = 0 if text_part in {"list", "table_row"} else OVERLAP_CHARS
    chunks = chunk_text(
        text,
        max_chars=LAW_MAX_CHARS,
        overlap_chars=overlap,
        min_chunk_chars=MIN_CHUNK_CHARS,
    )
    return build_chunk_records({**record, "text": text}, chunks, "law_sentence_split")
