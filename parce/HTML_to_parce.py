import re
from pathlib import Path

from bs4 import BeautifulSoup


IMPORTANT_SECTIONS = {
    "主文",
    "事実及び理由",
    "前段"
}

NOISE_SECTIONS = {
    "目次",
    "末尾事項",
    "附属書類"
}


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n\s*\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def detect_section_name(section_tag):
    """
    CSS判例HTMLの section 名抽出
    - class
    - id
    - 見出し（fallback）
    """

    # class優先
    if section_tag.get("class"):
        if isinstance(section_tag.get("class"), list):
            return section_tag.get("class")[0]
        return section_tag.get("class")

    # id
    if section_tag.get("id"):
        return section_tag.get("id")

    # 見出し系
    for h in ["h1", "h2", "h3"]:
        tag = section_tag.find(h)
        if tag:
            return tag.get_text(strip=True)

    return "unknown"


def extract_case_id(html_path):
    return Path(html_path).stem


def detect_title(soup):
    title_tag = soup.find("title")
    if title_tag:
        return normalize_text(title_tag.get_text(" ", strip=True))

    for selector in ["h1", "h2"]:
        tag = soup.find(selector)
        if tag:
            return normalize_text(tag.get_text(" ", strip=True))

    return ""


def iter_content_blocks(soup):
    sections = soup.find_all("section")
    if sections:
        yield from sections
        return

    main = soup.find("main")
    if main:
        yield main
        return

    body = soup.find("body")
    if body:
        yield body
        return

    yield soup


def parse_case_html(html_path):
    """
    判例HTML → RAG用レコード化
    """

    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    case_id = extract_case_id(html_path)

    title = detect_title(soup)

    records = []

    for sec_idx, sec in enumerate(iter_content_blocks(soup)):

        section_name = detect_section_name(sec)

        text = normalize_text(sec.get_text(separator="\n", strip=True))

        if not text:
            continue

        # ノイズセクションはフラグだけ残す（削除しない）
        is_noise = section_name in NOISE_SECTIONS
        is_important = section_name in IMPORTANT_SECTIONS

        # sectionが細かすぎる場合の安全対策。ただし主文などは短くても重要。
        if len(text) < 30 and not is_important:
            continue

        records.append({
            # ID系
            "case_id": case_id,
            "section_index": sec_idx,

            # メタ情報
            "title": title,
            "section": section_name,

            # 重要度
            "is_important": is_important,
            "is_noise": is_noise,

            # 本文（ここはまだチャンクしない）
            "text": text
        })

    return records
