from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable


TITLE_TAGS = {
    "ArticleTitle",
    "ArticleCaption",
    "ParagraphNum",
    "ItemTitle",
    "Subitem1Title",
    "Subitem2Title",
    "Subitem3Title",
    "Subitem4Title",
    "Subitem5Title",
    "Subitem6Title",
    "Subitem7Title",
    "Subitem8Title",
    "Subitem9Title",
    "Subitem10Title",
}

STRUCTURE_TITLE_TAGS = {
    "Part": "PartTitle",
    "Chapter": "ChapterTitle",
    "Section": "SectionTitle",
    "Subsection": "SubsectionTitle",
    "Division": "DivisionTitle",
    "SupplProvision": "SupplProvisionLabel",
    "AppdxTable": "AppdxTableTitle",
    "AppdxNote": "AppdxNoteTitle",
    "AppdxStyle": "AppdxStyleTitle",
    "AppdxFormat": "AppdxFormatTitle",
    "AppdxFig": "AppdxFigTitle",
}

LIST_TAG_RE = re.compile(r"^(Item|Subitem[1-9][0-9]*)$")


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def normalize_text(text: str | None) -> str:
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


def normalize_record_text(text: str | None) -> str:
    lines = [normalize_text(line) for line in (text or "").splitlines()]
    return "\n".join(line for line in lines if line)


def children_named(node: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in node if local_name(child.tag) == name]


def descendants_named(node: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in node.iter() if local_name(child.tag) == name]


def first_text(node: ET.Element, name: str, default: str = "") -> str:
    for child in node.iter():
        if local_name(child.tag) == name:
            text = node_text(child)
            if text:
                return text
    return default


def direct_text(node: ET.Element, name: str, default: str = "") -> str:
    for child in node:
        if local_name(child.tag) == name:
            text = node_text(child)
            if text:
                return text
    return default


def node_text(node: ET.Element) -> str:
    """Collect readable text, skipping ruby pronunciation nodes."""

    texts: list[str] = []

    def walk(current: ET.Element) -> None:
        if local_name(current.tag) == "Rt":
            return

        if current.text and current.text.strip():
            texts.append(current.text.strip())

        for child in current:
            walk(child)
            if child.tail and child.tail.strip():
                texts.append(child.tail.strip())

    walk(node)
    return normalize_text("".join(texts))


def get_sentences(node: ET.Element, *, skip_nested_lists: bool = False) -> str:
    texts: list[str] = []

    def walk(current: ET.Element, is_root: bool = False) -> None:
        name = local_name(current.tag)
        if skip_nested_lists and not is_root and LIST_TAG_RE.match(name):
            return

        if name == "Sentence":
            text = node_text(current)
            if text:
                texts.append(text)
            return

        for child in current:
            walk(child)

    walk(node, is_root=True)
    return "\n".join(texts)


def build_parent_map(root: ET.Element) -> dict[ET.Element, ET.Element]:
    return {child: parent for parent in root.iter() for child in parent}


def ancestors(
    node: ET.Element,
    parent_map: dict[ET.Element, ET.Element],
) -> Iterable[ET.Element]:
    parent = parent_map.get(node)
    while parent is not None:
        yield parent
        parent = parent_map.get(parent)


def has_ancestor(
    node: ET.Element,
    parent_map: dict[ET.Element, ET.Element],
    names: set[str],
) -> bool:
    return any(local_name(parent.tag) in names for parent in ancestors(node, parent_map))


def structure_context(
    node: ET.Element,
    parent_map: dict[ET.Element, ET.Element],
) -> dict[str, str]:
    context: dict[str, str] = {}

    for parent in reversed(list(ancestors(node, parent_map))):
        name = local_name(parent.tag)
        title_tag = STRUCTURE_TITLE_TAGS.get(name)
        if not title_tag:
            continue

        key = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
        if parent.get("Num"):
            context[f"{key}_num"] = parent.get("Num", "")

        title = direct_text(parent, title_tag, "")
        if title:
            context[f"{key}_title"] = title

    return context


def law_base_metadata(root: ET.Element, xml_path: Path) -> dict:
    return {
        "law_id": xml_path.stem,
        "law_name": first_text(root, "LawTitle", ""),
        "law_num": first_text(root, "LawNum", ""),
        "law_type": root.get("LawType"),
        "era": root.get("Era"),
        "year": root.get("Year"),
        "promulgate_month": root.get("PromulgateMonth"),
        "promulgate_day": root.get("PromulgateDay"),
    }


def make_record(base: dict, text: str, **metadata) -> dict | None:
    text = normalize_record_text(text)
    if not text:
        return None

    record = base.copy()
    record.update(metadata)
    record["text"] = text
    return record


def article_metadata(
    article: ET.Element,
    parent_map: dict[ET.Element, ET.Element],
) -> dict:
    metadata = structure_context(article, parent_map)
    metadata.update(
        {
            "article_num": article.get("Num"),
            "article_title": direct_text(article, "ArticleTitle", ""),
            "article_caption": direct_text(article, "ArticleCaption", ""),
        }
    )
    return metadata


def paragraph_sentence_node(paragraph: ET.Element) -> ET.Element:
    paragraph_sentences = children_named(paragraph, "ParagraphSentence")
    return paragraph_sentences[0] if paragraph_sentences else paragraph


def iter_list_records(
    base: dict,
    node: ET.Element,
    parent_metadata: dict,
    list_path: list[str] | None = None,
) -> Iterable[dict]:
    list_path = list_path or []

    for child in node:
        name = local_name(child.tag)
        if not LIST_TAG_RE.match(name):
            continue

        title = direct_text(child, f"{name}Title", "")
        num = child.get("Num")
        current_path = [*list_path, f"{name}:{num or title or ''}"]

        metadata = parent_metadata.copy()
        metadata.update(
            {
                "list_level": name,
                "list_num": num,
                "list_title": title,
                "list_path": "/".join(current_path),
            }
        )

        if name == "Item":
            metadata["item_num"] = num
            metadata["item_title"] = title
        else:
            metadata["subitem_level"] = name
            metadata["subitem_num"] = num
            metadata["subitem_title"] = title

        text = get_sentences(child, skip_nested_lists=True)
        record = make_record(base, text, text_part="list", **metadata)
        if record:
            yield record

        yield from iter_list_records(base, child, metadata, current_path)


def parse_article(
    base: dict,
    article: ET.Element,
    parent_map: dict[ET.Element, ET.Element],
) -> Iterable[dict]:
    metadata = article_metadata(article, parent_map)
    paragraphs = children_named(article, "Paragraph")

    if not paragraphs:
        record = make_record(
            base,
            get_sentences(article),
            text_part="article",
            **metadata,
        )
        if record:
            yield record
        return

    for paragraph in paragraphs:
        paragraph_metadata = metadata.copy()
        paragraph_metadata["paragraph_num"] = paragraph.get("Num")
        paragraph_metadata["paragraph_label"] = direct_text(paragraph, "ParagraphNum", "")

        paragraph_text = get_sentences(paragraph_sentence_node(paragraph))
        record = make_record(
            base,
            paragraph_text,
            text_part="paragraph",
            **paragraph_metadata,
        )
        if record:
            yield record

        yield from iter_list_records(base, paragraph, paragraph_metadata)


def parse_non_article_paragraph(
    base: dict,
    paragraph: ET.Element,
    parent_map: dict[ET.Element, ET.Element],
) -> Iterable[dict]:
    metadata = structure_context(paragraph, parent_map)
    metadata.update(
        {
            "article_num": None,
            "article_title": "",
            "article_caption": "",
            "paragraph_num": paragraph.get("Num"),
            "paragraph_label": direct_text(paragraph, "ParagraphNum", ""),
        }
    )

    record = make_record(
        base,
        get_sentences(paragraph_sentence_node(paragraph)),
        text_part="paragraph",
        **metadata,
    )
    if record:
        yield record

    yield from iter_list_records(base, paragraph, metadata)


def parse_table_row(
    base: dict,
    row: ET.Element,
    parent_map: dict[ET.Element, ET.Element],
) -> dict | None:
    cells = [
        get_sentences(cell) or node_text(cell)
        for cell in row
        if local_name(cell.tag) in {"TableColumn", "TableHeaderColumn"}
    ]
    text = " | ".join(cell for cell in cells if cell)

    return make_record(
        base,
        text,
        text_part="table_row",
        **structure_context(row, parent_map),
    )


def parse_law(xml_path):
    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    parent_map = build_parent_map(root)
    base = law_base_metadata(root, xml_path)

    records: list[dict] = []

    for article in descendants_named(root, "Article"):
        if has_ancestor(article, parent_map, {"TOC"}):
            continue
        records.extend(parse_article(base, article, parent_map))

    for paragraph in descendants_named(root, "Paragraph"):
        if has_ancestor(paragraph, parent_map, {"Article", "TOC"}):
            continue
        records.extend(parse_non_article_paragraph(base, paragraph, parent_map))

    for row in descendants_named(root, "TableRow"):
        if has_ancestor(row, parent_map, {"TOC"}):
            continue
        record = parse_table_row(base, row, parent_map)
        if record:
            records.append(record)

    return records
