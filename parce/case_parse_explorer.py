from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from bs4 import BeautifulSoup
from bs4.element import Tag

try:
    from chunck.chuncker import chunk_case_record
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(PROJECT_ROOT))
    try:
        from chunck.chuncker import chunk_case_record
    except ModuleNotFoundError:
        chunk_case_record = None


CASE_MAX_CHARS = 1200
OVERLAP_CHARS = 160
MIN_CHUNK_CHARS = 120


IMPORTANT_SECTIONS = {
    "主文",
    "事実",
    "理由",
    "事実及び理由",
    "争点",
    "当事者の主張",
    "裁判所の判断",
    "判断",
    "結論",
}

NOISE_LABELS = {
    "目次",
    "末尾事項",
    "附属書類",
    "原本",
    "正本",
}

CASE_METADATA_PATTERNS = {
    "court": re.compile(r"(最高裁判所|高等裁判所|地方裁判所|家庭裁判所|簡易裁判所|知的財産高等裁判所)"),
    "date": re.compile(r"((平成|令和|昭和|大正|明治)\s*\d+\s*年\s*\d+\s*月\s*\d+\s*日|\d{4}\s*年\s*\d+\s*月\s*\d+\s*日)"),
    "case_number": re.compile(r"(平成|令和|昭和|大正|明治)?\s*\d+\s*年\s*[\（(][^）)]{1,12}[\）)]\s*第?\s*\d+\s*号"),
}

SECTION_MARKER_RE = re.compile(
    r"^(主文|事実及び理由|事実|理由|争点|当事者の主張|裁判所の判断|判断|結論|第[一二三四五六七八九十\d]+)\s*$"
)
SENTENCE_RE = re.compile(r"(?<=[。．！？!?])")


COLAB_EXAMPLE = """\
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/llm_court
!pip install beautifulsoup4

!python parce/case_parse_explorer.py \\
  --cases /content/drive/MyDrive/path/to/hanrei_data \\
  --limit 20 \\
  --report-out /content/drive/MyDrive/llm_court_outputs/case_parse_report.json \\
  --best-jsonl-out /content/drive/MyDrive/llm_court_outputs/case_records_best.jsonl \\
  --best-chunks-out /content/drive/MyDrive/llm_court_outputs/cases.jsonl
"""


@dataclass(frozen=True)
class ParseStrategy:
    name: str
    description: str
    parser: Callable[[Path], list[dict]]


def normalize_text(text: str | None) -> str:
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t\u3000]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def compact_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def read_soup(html_path: Path) -> BeautifulSoup:
    raw = html_path.read_text(encoding="utf-8", errors="replace")
    return BeautifulSoup(raw, "html.parser")


def case_id_for(path: Path) -> str:
    return path.stem


def detect_title(soup: BeautifulSoup) -> str:
    for selector in ["title", "h1", "h2"]:
        tag = soup.find(selector)
        if tag:
            title = compact_text(tag.get_text(" ", strip=True))
            if title:
                return title
    return ""


def visible_root(soup: BeautifulSoup) -> Tag | BeautifulSoup:
    for name in ["main", "article", "body"]:
        tag = soup.find(name)
        if tag:
            return tag
    return soup


def base_record(html_path: Path, soup: BeautifulSoup) -> dict:
    text = normalize_text(visible_root(soup).get_text("\n", strip=True))
    metadata: dict[str, str] = {}

    for key, pattern in CASE_METADATA_PATTERNS.items():
        match = pattern.search(text)
        metadata[key] = compact_text(match.group(0)) if match else ""

    return {
        "case_id": case_id_for(html_path),
        "title": detect_title(soup),
        "source_type": "case",
        "source_path": str(html_path),
        **metadata,
    }


def section_name_from_tag(tag: Tag, fallback: str) -> str:
    classes = tag.get("class")
    if classes:
        if isinstance(classes, list):
            return compact_text(" ".join(str(value) for value in classes))
        return compact_text(str(classes))

    if tag.get("id"):
        return compact_text(str(tag.get("id")))

    for heading in tag.find_all(["h1", "h2", "h3"], recursive=False):
        text = compact_text(heading.get_text(" ", strip=True))
        if text:
            return text

    return fallback


def make_record(base: dict, section: str, text: str, section_index: int, strategy: str) -> dict | None:
    text = normalize_text(text)
    if not text:
        return None

    return {
        **base,
        "section": section,
        "section_index": section_index,
        "parse_strategy": strategy,
        "is_important": any(label in section for label in IMPORTANT_SECTIONS),
        "is_noise": any(label in section for label in NOISE_LABELS),
        "text": text,
    }


def parse_by_html_sections(html_path: Path) -> list[dict]:
    soup = read_soup(html_path)
    base = base_record(html_path, soup)
    root = visible_root(soup)
    section_tags = root.find_all("section") if isinstance(root, Tag) else []

    if not section_tags:
        section_tags = [tag for tag in root.find_all(["main", "article", "body"]) if isinstance(tag, Tag)]

    if not section_tags:
        section_tags = [root] if isinstance(root, Tag) else []

    records: list[dict] = []
    for index, tag in enumerate(section_tags):
        section = section_name_from_tag(tag, f"section_{index}")
        record = make_record(base, section, tag.get_text("\n", strip=True), index, "html_sections")
        if record and (len(record["text"]) >= 30 or record["is_important"]):
            records.append(record)
    return records


def parse_by_headings(html_path: Path) -> list[dict]:
    soup = read_soup(html_path)
    base = base_record(html_path, soup)
    root = visible_root(soup)
    headings = root.find_all(["h1", "h2", "h3", "h4"]) if isinstance(root, Tag) else []

    records: list[dict] = []
    current_heading = "本文"
    current_parts: list[str] = []

    def flush() -> None:
        nonlocal current_parts
        record = make_record(base, current_heading, "\n".join(current_parts), len(records), "headings")
        if record and len(record["text"]) >= 30:
            records.append(record)
        current_parts = []

    if not headings:
        record = make_record(base, "本文", root.get_text("\n", strip=True), 0, "headings")
        return [record] if record else []

    for node in root.descendants:
        if not isinstance(node, Tag):
            continue

        if node.name in {"script", "style", "nav", "footer"}:
            continue

        if node.name in {"h1", "h2", "h3", "h4"}:
            flush()
            current_heading = compact_text(node.get_text(" ", strip=True)) or current_heading
            continue

        if node.name in {"p", "li", "blockquote", "td", "th"}:
            text = normalize_text(node.get_text("\n", strip=True))
            if text:
                current_parts.append(text)

    flush()
    return records


def parse_by_legal_markers(html_path: Path) -> list[dict]:
    soup = read_soup(html_path)
    base = base_record(html_path, soup)
    text = normalize_text(visible_root(soup).get_text("\n", strip=True))
    records: list[dict] = []
    current_section = "本文"
    current_parts: list[str] = []

    def flush() -> None:
        nonlocal current_parts
        record = make_record(base, current_section, "\n".join(current_parts), len(records), "legal_markers")
        if record and (len(record["text"]) >= 30 or record["is_important"]):
            records.append(record)
        current_parts = []

    for line in text.splitlines():
        marker = SECTION_MARKER_RE.match(line)
        if marker:
            flush()
            current_section = marker.group(1)
            continue
        current_parts.append(line)

    flush()
    return records


def split_record_by_legal_markers(record: dict, strategy: str) -> list[dict]:
    text = normalize_text(record.get("text"))
    if not text:
        return []

    sections: list[tuple[str, str]] = []
    current_section = record.get("section") or "本文"
    current_parts: list[str] = []

    def flush() -> None:
        nonlocal current_parts
        section_text = "\n".join(current_parts)
        if normalize_text(section_text):
            sections.append((current_section, section_text))
        current_parts = []

    for line in text.splitlines():
        marker = SECTION_MARKER_RE.match(line)
        if marker:
            flush()
            marker_name = marker.group(1)
            parent_section = record.get("section") or "本文"
            current_section = marker_name if marker_name in parent_section else f"{parent_section} / {marker_name}"
            continue
        current_parts.append(line)

    flush()

    if len(sections) <= 1:
        return [{**record, "parse_strategy": strategy}]

    records: list[dict] = []
    for index, (section, section_text) in enumerate(sections):
        split_record = make_record(record, section, section_text, index, strategy)
        if split_record and (len(split_record["text"]) >= 30 or split_record["is_important"]):
            records.append(split_record)
    return records


def parse_by_headings_legal_hybrid(html_path: Path) -> list[dict]:
    records: list[dict] = []

    for record in parse_by_headings(html_path):
        marker_splits = split_record_by_legal_markers(record, "headings_legal_hybrid")
        if len(marker_splits) > 1:
            records.extend(marker_splits)
            continue

        records.append({**record, "parse_strategy": "headings_legal_hybrid"})

    return records


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for line in normalize_text(text).splitlines():
        parts = [part.strip() for part in SENTENCE_RE.split(line) if part.strip()]
        sentences.extend(parts or [line])
    return sentences


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

        sentence_parts = [part.strip() for part in SENTENCE_RE.split(line) if part.strip()]
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
    max_chars: int = CASE_MAX_CHARS,
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
        separator_len = 1 if current else 0
        unit_len = len(unit)

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

    if len(chunks) >= 2 and len(chunks[-1]) < min_chunk_chars:
        merged = f"{chunks[-2]}\n{chunks[-1]}".strip()
        if len(merged) <= max_chars:
            chunks = [*chunks[:-2], merged]

    return chunks


def fallback_chunk_case_record(record: dict) -> list[dict]:
    text = normalize_text(record.get("text"))
    section = record.get("section", "")
    if not text:
        return []

    chunks = [text] if "主文" in section else chunk_text(text)
    chunk_records: list[dict] = []

    for index, chunk in enumerate(chunks):
        chunk_records.append(
            {
                **record,
                "text": chunk,
                "chunk_index": index,
                "chunk_count": len(chunks),
                "chunk_strategy": "case_parse_explorer_fallback",
                "chunk_char_count": len(chunk),
            }
        )

    return chunk_records


def build_case_chunks(record: dict) -> list[dict]:
    if chunk_case_record is not None:
        return chunk_case_record(record)
    return fallback_chunk_case_record(record)


def parse_by_sentence_windows(html_path: Path) -> list[dict]:
    soup = read_soup(html_path)
    base = base_record(html_path, soup)
    text = normalize_text(visible_root(soup).get_text("\n", strip=True))
    sentences = split_sentences(text)
    records: list[dict] = []
    window_size = 8
    step = 6

    for start in range(0, len(sentences), step):
        window = sentences[start : start + window_size]
        if not window:
            continue

        record = make_record(
            base,
            f"sentence_window_{start // step}",
            "\n".join(window),
            len(records),
            "sentence_windows",
        )
        if record and len(record["text"]) >= 80:
            records.append(record)
    return records


def parse_by_dom_blocks(html_path: Path) -> list[dict]:
    soup = read_soup(html_path)
    base = base_record(html_path, soup)
    root = visible_root(soup)
    block_tags = root.find_all(["section", "div", "p", "li", "blockquote"]) if isinstance(root, Tag) else []
    records: list[dict] = []
    seen: set[str] = set()

    for tag in block_tags:
        if tag.find(["section", "div", "p", "li", "blockquote"]):
            continue

        text = normalize_text(tag.get_text("\n", strip=True))
        fingerprint = compact_text(text)
        if len(fingerprint) < 40 or fingerprint in seen:
            continue
        seen.add(fingerprint)

        section = section_name_from_tag(tag, f"block_{len(records)}")
        record = make_record(base, section, text, len(records), "dom_blocks")
        if record:
            records.append(record)

    if records:
        return records

    record = make_record(base, "本文", root.get_text("\n", strip=True), 0, "dom_blocks")
    return [record] if record else []


def candidate_strategies() -> list[ParseStrategy]:
    return [
        ParseStrategy("html_sections", "HTMLのsection/main/article/body単位を尊重する", parse_by_html_sections),
        ParseStrategy("headings", "h1-h4見出し配下で本文をまとめる", parse_by_headings),
        ParseStrategy("headings_legal_hybrid", "h1-h4見出しで分けた後に主文・理由などの法的見出しで再分割する", parse_by_headings_legal_hybrid),
        ParseStrategy("legal_markers", "主文・理由などの法的見出し行で分割する", parse_by_legal_markers),
        ParseStrategy("sentence_windows", "文単位の固定窓でRAG向け粒度を安定させる", parse_by_sentence_windows),
        ParseStrategy("dom_blocks", "末端DOMブロック単位で細かく分割する", parse_by_dom_blocks),
    ]


def iter_html_paths(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() in {".html", ".htm"}:
            yield input_path
        return

    if not input_path.exists():
        raise FileNotFoundError(f"input path does not exist: {input_path}")

    for path in sorted(input_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".html", ".htm"}:
            yield path


def select_html_paths(
    input_path: Path,
    *,
    limit: int | None = None,
    sample: str | None = None,
    sample_size: int | None = None,
) -> list[Path]:
    html_paths = list(iter_html_paths(input_path))

    if sample:
        size = sample_size or limit or 500
        if sample == "first":
            html_paths = html_paths[:size]
        elif sample == "middle":
            start = max((len(html_paths) - size) // 2, 0)
            html_paths = html_paths[start : start + size]
        elif sample == "last":
            html_paths = html_paths[-size:]
        else:
            raise ValueError(f"unknown sample: {sample}")
    elif limit is not None:
        html_paths = html_paths[:limit]

    return html_paths


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_records(records: list[dict]) -> dict:
    if not records:
        return {
            "score": 0.0,
            "record_count": 0,
            "chunk_count": 0,
            "avg_chars": 0.0,
            "median_chars": 0.0,
            "important_section_ratio": 0.0,
            "metadata_fill_ratio": 0.0,
            "noise_ratio": 1.0,
            "duplicate_ratio": 1.0,
            "chunk_size_fit_ratio": 0.0,
        }

    chunks: list[dict] = []
    for record in records:
        chunks.extend(build_case_chunks(record))

    lengths = [len(record["text"]) for record in records]
    fingerprints = [compact_text(record["text"]) for record in records]
    unique_count = len(set(fingerprints))
    important_count = sum(1 for record in records if record.get("is_important"))
    noise_count = sum(1 for record in records if record.get("is_noise"))
    metadata_keys = ["title", "court", "date", "case_number"]
    metadata_filled = sum(1 for key in metadata_keys if records[0].get(key))
    fit_count = sum(1 for chunk in chunks if 180 <= len(chunk["text"]) <= 1200)

    important_ratio = safe_ratio(important_count, len(records))
    metadata_ratio = safe_ratio(metadata_filled, len(metadata_keys))
    noise_ratio = safe_ratio(noise_count, len(records))
    duplicate_ratio = 1.0 - safe_ratio(unique_count, len(records))
    chunk_fit_ratio = safe_ratio(fit_count, len(chunks))
    count_balance = min(1.0, safe_ratio(len(records), 8.0)) * min(1.0, safe_ratio(80.0, len(records)))
    avg_chars = statistics.mean(lengths)
    median_chars = statistics.median(lengths)
    length_balance = 1.0 if 250 <= median_chars <= 2500 else 0.4 if 100 <= median_chars <= 5000 else 0.1

    score = (
        0.22 * metadata_ratio
        + 0.20 * chunk_fit_ratio
        + 0.18 * length_balance
        + 0.14 * count_balance
        + 0.12 * min(1.0, important_ratio * 4)
        + 0.08 * (1.0 - noise_ratio)
        + 0.06 * (1.0 - duplicate_ratio)
    )

    return {
        "score": round(score, 4),
        "record_count": len(records),
        "chunk_count": len(chunks),
        "avg_chars": round(avg_chars, 1),
        "median_chars": round(median_chars, 1),
        "important_section_ratio": round(important_ratio, 4),
        "metadata_fill_ratio": round(metadata_ratio, 4),
        "noise_ratio": round(noise_ratio, 4),
        "duplicate_ratio": round(duplicate_ratio, 4),
        "chunk_size_fit_ratio": round(chunk_fit_ratio, 4),
    }


def aggregate_metrics(per_file_metrics: list[dict]) -> dict:
    if not per_file_metrics:
        return evaluate_records([])

    keys = per_file_metrics[0].keys()
    aggregated = {}
    for key in keys:
        values = [metrics[key] for metrics in per_file_metrics]
        aggregated[key] = round(statistics.mean(values), 4)
    return aggregated


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def explore(
    input_path: Path,
    *,
    limit: int | None = None,
    sample: str | None = None,
    sample_size: int | None = None,
) -> dict:
    html_paths = select_html_paths(input_path, limit=limit, sample=sample, sample_size=sample_size)
    if not html_paths:
        raise ValueError(f"no HTML files found under: {input_path}")

    strategy_results = []
    for strategy in candidate_strategies():
        per_file = []
        errors = []

        for html_path in html_paths:
            try:
                records = strategy.parser(html_path)
                metrics = evaluate_records(records)
            except Exception as exc:
                errors.append({"path": str(html_path), "error": str(exc)})
                metrics = evaluate_records([])

            per_file.append({"path": str(html_path), **metrics})

        aggregate = aggregate_metrics([{key: value for key, value in metrics.items() if key != "path"} for metrics in per_file])
        strategy_results.append(
            {
                "strategy": strategy.name,
                "description": strategy.description,
                "aggregate": aggregate,
                "errors": errors,
                "files": per_file,
            }
        )

    strategy_results.sort(key=lambda item: item["aggregate"]["score"], reverse=True)
    return {
        "input_path": str(input_path),
        "sample": sample or ("limit" if limit is not None else "all"),
        "sample_size": sample_size,
        "file_count": len(html_paths),
        "best_strategy": strategy_results[0]["strategy"],
        "strategies": strategy_results,
    }


def records_for_strategy(
    strategy_name: str,
    input_path: Path,
    limit: int | None,
    sample: str | None = None,
    sample_size: int | None = None,
) -> Iterable[dict]:
    strategies = {strategy.name: strategy for strategy in candidate_strategies()}
    strategy = strategies[strategy_name]

    for html_path in select_html_paths(input_path, limit=limit, sample=sample, sample_size=sample_size):
        records = strategy.parser(html_path)
        for record in records:
            yield record


def chunked_records(records: Iterable[dict]) -> Iterable[dict]:
    for record in records:
        yield from build_case_chunks(record)


def run_explorer(
    cases: str | Path,
    *,
    limit: int | None = None,
    sample: str | None = None,
    sample_size: int | None = None,
    report_out: str | Path | None = None,
    best_jsonl_out: str | Path | None = None,
    best_chunks_out: str | Path | None = None,
) -> dict:
    """Notebook-friendly entry point, useful when case data lives on Google Drive."""

    cases_path = Path(cases)
    report = explore(cases_path, limit=limit, sample=sample, sample_size=sample_size)
    best = report["best_strategy"]
    best_records = list(
        records_for_strategy(best, cases_path, limit, sample=sample, sample_size=sample_size)
    )

    if report_out:
        report_path = Path(report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if best_jsonl_out:
        write_jsonl(Path(best_jsonl_out), best_records)

    if best_chunks_out:
        write_jsonl(Path(best_chunks_out), chunked_records(best_records))

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore multiple case HTML parse designs and rank them for RAG preprocessing."
    )
    parser.add_argument("--cases", type=Path, help="HTML file or directory containing case HTML files.")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N HTML files.")
    parser.add_argument("--sample", choices=["first", "middle", "last"], help="Evaluate a positional sample.")
    parser.add_argument("--sample-size", type=int, default=500, help="Number of HTML files to use with --sample.")
    parser.add_argument("--report-out", type=Path, help="Write the strategy comparison report as JSON.")
    parser.add_argument("--best-jsonl-out", type=Path, help="Write records from the best strategy as JSONL.")
    parser.add_argument("--best-chunks-out", type=Path, help="Write chunked records from the best strategy as JSONL.")
    parser.add_argument("--print-colab-example", action="store_true", help="Print a Google Colab usage example and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.print_colab_example:
        print(COLAB_EXAMPLE)
        return

    if not args.cases:
        raise SystemExit("--cases is required unless --print-colab-example is used.")

    report = run_explorer(
        args.cases,
        limit=args.limit,
        sample=args.sample,
        sample_size=args.sample_size,
        report_out=args.report_out,
        best_jsonl_out=args.best_jsonl_out,
        best_chunks_out=args.best_chunks_out,
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
