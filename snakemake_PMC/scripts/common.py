"""Shared schemas and JATS parsing helpers for the PMC corpus workflow."""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
from lxml import etree


CORPUS_SCHEMA = pa.schema(
    [
        pa.field("pmcid", pa.string(), nullable=False),
        pa.field("article_version", pa.string(), nullable=False),
        pa.field("section", pa.string(), nullable=False),
        pa.field("paragraph", pa.int32(), nullable=False),
        pa.field("sentence_range", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
    ]
)

ARTICLE_SCHEMA = pa.schema(
    [
        pa.field("pmcid", pa.string(), nullable=False),
        pa.field("article_version", pa.string(), nullable=False),
        pa.field("version", pa.int32(), nullable=False),
        pa.field("pmid", pa.string()),
        pa.field("doi", pa.string()),
        pa.field("title", pa.string()),
        pa.field("citation", pa.string()),
        pa.field("journal", pa.string()),
        pa.field("publication_year", pa.int32()),
        pa.field("article_language", pa.string()),
        pa.field("license_code", pa.string()),
        pa.field("article_type", pa.string()),
        pa.field("is_pmc_openaccess", pa.bool_()),
        pa.field("is_retracted", pa.bool_()),
        pa.field("included", pa.bool_(), nullable=False),
        pa.field("exclusion_reason", pa.string()),
        pa.field("inventory_last_modified", pa.string()),
        pa.field("source_etag", pa.string()),
        pa.field("xml_url", pa.string(), nullable=False),
        pa.field("snapshot_date", pa.string(), nullable=False),
        pa.field("paragraphs", pa.int32(), nullable=False),
        pa.field("corpus_rows", pa.int32(), nullable=False),
    ]
)

MANIFEST_SCHEMA = pa.schema(
    [
        pa.field("pmcid", pa.string(), nullable=False),
        pa.field("pmcid_num", pa.int64(), nullable=False),
        pa.field("version", pa.int32(), nullable=False),
        pa.field("article_version", pa.string(), nullable=False),
        pa.field("inventory_last_modified", pa.string()),
        pa.field("source_etag", pa.string()),
        pa.field("metadata_key", pa.string(), nullable=False),
        pa.field("metadata_url", pa.string()),
        pa.field("xml_url", pa.string(), nullable=False),
        pa.field("snapshot_date", pa.string(), nullable=False),
    ]
)

INVENTORY_COLUMNS = (
    "bucket",
    "metadata_key",
    "last_modified",
    "etag",
)

XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
BLOCKED_TEXT_TAGS = {
    "fig",
    "graphic",
    "inline-graphic",
    "media",
    "supplementary-material",
    "table-wrap",
    "table",
    "disp-formula",
    "inline-formula",
    "tex-math",
}

SPACE_RE = re.compile(r"\s+")
CC_LICENSE_RE = re.compile(
    r"\b(CC0|CC\s+BY(?:-NC)?(?:-ND|-SA)?(?:-NC-(?:ND|SA))?)\b", re.IGNORECASE
)
YEAR_RE = re.compile(r"\b(18|19|20|21)\d{2}\b")

# Protect common scientific abbreviations before applying deterministic sentence
# boundary rules. This keeps the corpus build independent of downloaded NLP data.
ABBREVIATIONS = (
    "e.g.",
    "i.e.",
    "et al.",
    "etc.",
    "Fig.",
    "Figs.",
    "Eq.",
    "Eqs.",
    "Ref.",
    "Refs.",
    "Dr.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "Prof.",
    "vs.",
    "cf.",
    "sp.",
    "spp.",
    "subsp.",
    "var.",
    "no.",
    "vol.",
)
SENTINEL = "\ue000"


@dataclass(frozen=True)
class ParseOptions:
    max_text_chars: int = 512
    min_text_chars: int = 20
    publication_year_min: int = 1950
    allowed_languages: tuple[str, ...] = ("en",)
    include_unknown_language: bool = True
    excluded_section_patterns: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ParseOptions":
        return cls(
            max_text_chars=int(values.get("max_text_chars", 512)),
            min_text_chars=int(values.get("min_text_chars", 20)),
            publication_year_min=int(values.get("publication_year_min", 1950)),
            allowed_languages=tuple(
                str(item).lower() for item in values.get("allowed_languages", ["en"])
            ),
            include_unknown_language=bool(
                values.get("include_unknown_language", True)
            ),
            excluded_section_patterns=tuple(
                str(item) for item in values.get("excluded_section_patterns", [])
            ),
        )


def normalize_space(value: str | None) -> str:
    if not value:
        return ""
    return SPACE_RE.sub(" ", value).strip()


def local_name(element: etree._Element) -> str:
    return etree.QName(element).localname


def element_text(
    element: etree._Element, blocked_tags: set[str] | None = None
) -> str:
    """Extract readable element text while omitting figures, tables and formulas."""

    blocked = BLOCKED_TEXT_TAGS if blocked_tags is None else blocked_tags
    parts: list[str] = []

    def visit(node: etree._Element) -> None:
        if node.text:
            parts.append(node.text)
        for child in node:
            if local_name(child) not in blocked:
                visit(child)
            if child.tail:
                parts.append(child.tail)

    visit(element)
    return normalize_space(" ".join(parts))


def first_text(root: etree._Element, xpath: str) -> str | None:
    matches = root.xpath(xpath)
    for match in matches:
        if isinstance(match, etree._Element):
            value = element_text(match)
        else:
            value = normalize_space(str(match))
        if value:
            return value
    return None


def article_identifier(article: etree._Element, *identifier_types: str) -> str | None:
    wanted = {value.lower() for value in identifier_types}
    for element in article.xpath(".//*[local-name()='article-id']"):
        identifier_type = (
            element.get("pub-id-type") or element.get("article-id-type") or ""
        ).lower()
        if identifier_type in wanted:
            value = element_text(element)
            if value:
                return value
    return None


def publication_year(article: etree._Element) -> int | None:
    date_priority = ("epub", "ppub", "collection", "pub", "")
    dates = article.xpath(".//*[local-name()='article-meta']/*[local-name()='pub-date']")
    for preferred in date_priority:
        for date in dates:
            date_type = (date.get("pub-type") or date.get("date-type") or "").lower()
            if preferred and date_type != preferred:
                continue
            value = first_text(date, "./*[local-name()='year']")
            if value and value.isdigit():
                return int(value)
    citation = first_text(article, ".//*[local-name()='article-meta']")
    match = YEAR_RE.search(citation or "")
    return int(match.group(0)) if match else None


def normalized_license(article: etree._Element) -> str | None:
    license_elements = article.xpath(
        ".//*[local-name()='article-meta']//*[local-name()='license']"
    )
    for element in license_elements:
        license_type = normalize_space(element.get("license-type"))
        href = normalize_space(
            element.get("{http://www.w3.org/1999/xlink}href")
            or element.get("href")
        )
        text = element_text(element)
        combined = " ".join(value for value in (license_type, href, text) if value)
        lowered = combined.lower()
        if "creativecommons.org/publicdomain/zero" in lowered or re.search(
            r"\bcc\s*0\b", combined, re.IGNORECASE
        ):
            return "CC0"
        match = CC_LICENSE_RE.search(combined.replace("_", "-").replace("/", " "))
        if match:
            return SPACE_RE.sub(" ", match.group(1).upper())
        if "text and data mining" in lowered or license_type.upper() == "TDM":
            return "TDM"
    return None


def split_sentences(text: str) -> list[str]:
    """Split biomedical prose without external model downloads.

    This is deliberately conservative: false merges are safer than splitting
    strain names such as ``E. coli`` in the middle of a sentence.
    """

    protected = normalize_space(text)
    if not protected:
        return []

    for abbreviation in ABBREVIATIONS:
        protected = re.sub(
            re.escape(abbreviation),
            lambda match: match.group(0).replace(".", SENTINEL),
            protected,
            flags=re.IGNORECASE,
        )

    # Protect single-letter genus abbreviations before lowercase species names.
    protected = re.sub(
        r"\b([A-Z])\.\s+(?=[a-z][a-z-]+\b)",
        rf"\1{SENTINEL} ",
        protected,
    )

    boundaries = re.split(r"(?<=[.!?])\s+(?=[\"'\(\[]?[A-Z0-9])", protected)
    sentences = [normalize_space(item.replace(SENTINEL, ".")) for item in boundaries]
    return [item for item in sentences if item]


def pack_sentences(sentences: Sequence[str], max_chars: int) -> Iterable[tuple[str, str]]:
    """Pack adjacent sentences up to ``max_chars`` and retain their 1-based range."""

    current: list[str] = []
    start = 1
    for number, sentence in enumerate(sentences, start=1):
        proposed_length = sum(len(item) for item in current) + len(current) + len(sentence)
        if current and proposed_length > max_chars:
            end = number - 1
            yield (str(start) if start == end else f"{start}-{end}", " ".join(current))
            current = []
            start = number
        current.append(sentence)
    if current:
        end = start + len(current) - 1
        yield (str(start) if start == end else f"{start}-{end}", " ".join(current))


def section_path(paragraph: etree._Element) -> str:
    sections = paragraph.xpath("ancestor::*[local-name()='sec']")
    titles: list[str] = []
    for section in sections:
        title = first_text(section, "./*[local-name()='title']")
        if title and title not in titles:
            titles.append(title)
    return " > ".join(titles) if titles else "body"


def is_excluded_section(section: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(pattern.search(section) for pattern in patterns)


def _language(article: etree._Element) -> str | None:
    value = article.get(XML_LANG) or article.get("lang")
    if not value:
        value = first_text(
            article,
            ".//*[local-name()='article-meta']/*[local-name()='custom-meta-group']"
            "/*[local-name()='custom-meta'][*[local-name()='meta-name' and "
            "contains(translate(text(), 'LANGUAGE', 'language'), 'language')]]"
            "/*[local-name()='meta-value']",
        )
    if not value:
        return None
    return value.lower().replace("_", "-").split("-")[0]


def _abstract_paragraphs(article: etree._Element) -> list[tuple[str, etree._Element]]:
    results: list[tuple[str, etree._Element]] = []
    abstracts = article.xpath(
        ".//*[local-name()='article-meta']/*[local-name()='abstract']"
    )
    for abstract_index, abstract in enumerate(abstracts, start=1):
        abstract_title = first_text(abstract, "./*[local-name()='title']")
        base = abstract_title or ("abstract" if abstract_index == 1 else f"abstract {abstract_index}")
        paragraphs = abstract.xpath(".//*[local-name()='p']")
        if paragraphs:
            for paragraph in paragraphs:
                if paragraph.xpath(
                    "ancestor::*[local-name()='p' or local-name()='fig' or "
                    "local-name()='table-wrap' or local-name()='table' or "
                    "local-name()='supplementary-material']"
                ):
                    continue
                parent_sections = paragraph.xpath("ancestor::*[local-name()='sec']")
                nested_titles = [
                    first_text(section, "./*[local-name()='title']")
                    for section in parent_sections
                ]
                nested_titles = [title for title in nested_titles if title]
                section = " > ".join([base, *nested_titles]) if nested_titles else base
                results.append((section, paragraph))
        else:
            results.append((base, abstract))
    return results


def parse_jats(
    xml_bytes: bytes,
    manifest: Mapping[str, Any],
    options: ParseOptions,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parser = etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        load_dtd=False,
        recover=False,
        huge_tree=True,
        remove_comments=True,
    )
    root = etree.fromstring(xml_bytes, parser=parser)
    article_matches = root.xpath("self::*[local-name()='article'] | .//*[local-name()='article']")
    if not article_matches:
        raise ValueError("XML contains no JATS article element")
    article = article_matches[0]

    pmcid = str(manifest["pmcid"])
    article_version = str(manifest["article_version"])
    version = int(manifest["version"])
    title = first_text(
        article,
        ".//*[local-name()='article-meta']/*[local-name()='title-group']"
        "/*[local-name()='article-title']",
    )
    journal = first_text(
        article,
        ".//*[local-name()='journal-meta']//*[local-name()='journal-title']",
    )
    year = publication_year(article)
    language = _language(article)

    exclusion_reason: str | None = None
    if year is not None and year < options.publication_year_min:
        exclusion_reason = f"publication_year_before_{options.publication_year_min}"
    elif language and options.allowed_languages and language not in options.allowed_languages:
        exclusion_reason = f"language_{language}"
    elif not language and not options.include_unknown_language:
        exclusion_reason = "language_unknown"

    excluded_patterns = [
        re.compile(pattern, re.IGNORECASE) for pattern in options.excluded_section_patterns
    ]
    rows: list[dict[str, Any]] = []
    paragraph_number = 0

    def add_paragraph(section: str, text: str) -> None:
        nonlocal paragraph_number
        # Number source blocks before filtering so provenance does not shift when
        # a section filter or minimum-length threshold changes.
        paragraph_number += 1
        cleaned = normalize_space(text)
        if len(cleaned) < options.min_text_chars:
            return
        if is_excluded_section(section, excluded_patterns):
            return
        sentences = split_sentences(cleaned)
        if not sentences:
            return
        for sentence_range, packed_text in pack_sentences(
            sentences, options.max_text_chars
        ):
            if len(packed_text) < options.min_text_chars:
                continue
            rows.append(
                {
                    "pmcid": pmcid,
                    "article_version": article_version,
                    "section": normalize_space(section).lower() or "body",
                    "paragraph": paragraph_number,
                    "sentence_range": sentence_range,
                    "text": packed_text,
                }
            )

    if exclusion_reason is None:
        if title:
            add_paragraph("title", title)
        for section, paragraph in _abstract_paragraphs(article):
            add_paragraph(section, element_text(paragraph))
        body_paragraphs = article.xpath(
            "./*[local-name()='body']//*[local-name()='p']"
        )
        for paragraph in body_paragraphs:
            # Nested paragraphs and captions/tables would otherwise be emitted
            # twice: once as part of surrounding prose and once as their own row.
            if paragraph.xpath(
                "ancestor::*[local-name()='p' or local-name()='fig' or "
                "local-name()='table-wrap' or local-name()='table' or "
                "local-name()='supplementary-material']"
            ):
                continue
            add_paragraph(section_path(paragraph), element_text(paragraph))

    article_record = {
        "pmcid": pmcid,
        "article_version": article_version,
        "version": version,
        "pmid": article_identifier(article, "pmid"),
        "doi": article_identifier(article, "doi"),
        "title": title,
        "citation": None,
        "journal": journal,
        "publication_year": year,
        "article_language": language,
        "license_code": normalized_license(article),
        "article_type": article.get("article-type"),
        "is_pmc_openaccess": None,
        "is_retracted": None,
        "included": exclusion_reason is None,
        "exclusion_reason": exclusion_reason,
        "inventory_last_modified": manifest.get("inventory_last_modified"),
        "source_etag": manifest.get("source_etag"),
        "xml_url": str(manifest["xml_url"]),
        "snapshot_date": str(manifest["snapshot_date"]),
        "paragraphs": paragraph_number,
        "corpus_rows": len(rows),
    }
    return article_record, rows


def empty_table(schema: pa.Schema) -> pa.Table:
    return pa.Table.from_pylist([], schema=schema)


def records_table(records: Sequence[Mapping[str, Any]], schema: pa.Schema) -> pa.Table:
    if not records:
        return empty_table(schema)
    return pa.Table.from_pylist(list(records), schema=schema)


def redirect_snakemake_log(smk: Any) -> None:
    """Send script stdout/stderr to the rule's declared log on cluster jobs."""

    if not getattr(smk, "log", None):
        return
    log_path = Path(str(smk.log[0]))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = handle
    sys.stderr = handle
