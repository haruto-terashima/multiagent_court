from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

from parce.HTML_to_parce import parse_case_html
from parce.XML_to_parce import parse_law
from chunck.chuncker import chunk_case_record, chunk_law_record


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            if not record.get("text"):
                continue
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def iter_paths(input_path: Path, suffixes: Sequence[str]) -> Iterable[Path]:
    suffixes = tuple(s.lower() for s in suffixes)

    if input_path.is_file():
        if input_path.suffix.lower() in suffixes:
            yield input_path
        return

    if not input_path.exists():
        print(f"[WARN] input path does not exist: {input_path}", file=sys.stderr)
        return

    for path in sorted(input_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes:
            yield path


def iter_case_chunks(input_path: Path) -> Iterable[dict]:
    for html_path in iter_paths(input_path, [".html", ".htm"]):
        try:
            records = parse_case_html(html_path)
        except Exception as exc:
            print(f"[ERROR] failed to parse case HTML: {html_path} -> {exc}", file=sys.stderr)
            continue

        for record in records:
            record.setdefault("source_type", "case")
            record.setdefault("source_path", str(html_path))
            yield from chunk_case_record(record)


def iter_law_chunks(input_path: Path) -> Iterable[dict]:
    for xml_path in iter_paths(input_path, [".xml"]):
        try:
            records = parse_law(xml_path)
        except Exception as exc:
            print(f"[ERROR] failed to parse law XML: {xml_path} -> {exc}", file=sys.stderr)
            continue

        for record in records:
            record.setdefault("source_type", "law")
            record.setdefault("source_path", str(xml_path))
            yield from chunk_law_record(record)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert court HTML/XML files into chunked JSONL for RAG.")
    parser.add_argument("--cases", type=Path, help="HTML file or directory containing case HTML files.")
    parser.add_argument("--laws", type=Path, help="XML file or directory containing law XML files.")
    parser.add_argument("--out-dir", type=Path, default=Path("data_after_parce/chunked"))
    args = parser.parse_args()

    if not args.cases and not args.laws:
        parser.error("At least one of --cases or --laws is required.")

    if args.cases:
        count = write_jsonl(args.out_dir / "cases.jsonl", iter_case_chunks(args.cases))
        print(f"wrote {count} case chunks to {args.out_dir / 'cases.jsonl'}")

    if args.laws:
        count = write_jsonl(args.out_dir / "laws.jsonl", iter_law_chunks(args.laws))
        print(f"wrote {count} law chunks to {args.out_dir / 'laws.jsonl'}")


if __name__ == "__main__":
    main()
