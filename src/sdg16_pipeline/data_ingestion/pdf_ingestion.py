import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def safe_document_id(path: Path) -> str:
    return normalize_key(path.stem)


GROUP_METADATA = {
    "nhom_1": {
        "group": "group_1_imd_wef_reports",
        "group_label": "Nhóm 1",
        "source_type": "competitiveness_report",
        "collection_role": "IMD & WEF Reports",
    },
    "nhom_2": {
        "group": "group_2_vietnam_country_reports",
        "group_label": "Nhóm 2",
        "source_type": "country_report",
        "collection_role": "Vietnam Country Reports",
        "country": "Vietnam",
    },
    "nhom_3": {
        "group": "group_3_country_case_studies",
        "group_label": "Nhóm 3",
        "source_type": "case_study",
        "collection_role": "Country Case Studies",
    },
    "nhom_4": {
        "group": "group_4_scientific_papers",
        "group_label": "Nhóm 4",
        "source_type": "scientific_paper",
        "collection_role": "Scientific Papers",
    },
    "nhom_5": {
        "group": "group_5_vietnam_policies",
        "group_label": "Nhóm 5",
        "source_type": "policy_document",
        "collection_role": "Vietnam Policies",
        "country": "Vietnam",
    },
}

GROUP_EXPORT_ORDER = [
    "group_1_imd_wef_reports",
    "group_2_vietnam_country_reports",
    "group_3_country_case_studies",
    "group_4_scientific_papers",
    "group_5_vietnam_policies",
]

GROUP_EXPORT_TITLES = {
    "group_1_imd_wef_reports": "NHÓM 1 — IMD & WEF Reports",
    "group_2_vietnam_country_reports": "NHÓM 2 — Vietnam Country Reports",
    "group_3_country_case_studies": "NHÓM 3 — Case Studies nước tăng hạng IMD nhanh",
    "group_4_scientific_papers": "NHÓM 4 — Academic Papers",
    "group_5_vietnam_policies": "NHÓM 5 — Chính sách Việt Nam",
}


def strip_vietnamese_marks(value: str) -> str:
    replacements = {
        "đ": "d",
        "Đ": "D",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    try:
        import unicodedata

        value = "".join(
            char
            for char in unicodedata.normalize("NFD", value)
            if unicodedata.category(char) != "Mn"
        )
    except Exception:
        pass
    return value


def group_key(value: str) -> str:
    return normalize_key(strip_vietnamese_marks(value))


def infer_country_from_filename(path: Path) -> str | None:
    key = group_key(path.stem)
    country_tokens = {
        "singapore": "Singapore",
        "thailan": "Thailand",
        "thai_lan": "Thailand",
        "malaysia": "Malaysia",
        "hanquoc": "Korea",
        "han_quoc": "Korea",
        "korea": "Korea",
        "ireland": "Ireland",
        "vietnam": "Vietnam",
        "viet_nam": "Vietnam",
        "pci": "Vietnam",
        "luat": "Vietnam",
        "nghiquyet": "Vietnam",
        "nghi_quyet": "Vietnam",
    }
    for token, country in country_tokens.items():
        if token in key:
            return country
    return None


def infer_metadata_from_path(path: Path, input_dir: Path) -> dict[str, Any]:
    try:
        relative_parts = path.relative_to(input_dir).parts
    except ValueError:
        relative_parts = path.parts

    inferred: dict[str, Any] = {
        "document_id": safe_document_id(path),
        "title": path.stem,
        "manifest_status": "inferred_from_local_folder",
        "source_type": "pdf",
    }

    for part in relative_parts[:-1]:
        key = group_key(part)
        if key in GROUP_METADATA:
            inferred.update(GROUP_METADATA[key])
            break

    if "group" not in inferred:
        inferred.update(GROUP_METADATA["nhom_5"])

    country = infer_country_from_filename(path)
    if country and "country" not in inferred:
        inferred["country"] = country
    return inferred


@dataclass
class ParsedElement:
    text: str
    metadata: dict[str, Any]


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not str(path) or not path.exists() or path.is_dir():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    documents = payload.get("documents", [])
    lookup: dict[str, dict[str, Any]] = {}
    for document in documents:
        filename = document.get("filename")
        document_id = document.get("document_id")
        keys = []
        if filename:
            keys.extend([normalize_key(filename), normalize_key(Path(filename).stem)])
        if document_id:
            keys.append(normalize_key(document_id))
        for key in keys:
            lookup[key] = document
    return lookup


def find_manifest_metadata(path: Path, manifest: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        normalize_key(path.name),
        normalize_key(path.stem),
    ]
    for candidate in candidates:
        if candidate in manifest:
            return dict(manifest[candidate])
    return {}


def parse_with_pypdf(path: Path) -> list[ParsedElement]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parsed: list[ParsedElement] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        parsed.append(
            ParsedElement(
                text=text,
                metadata={
                    "page_number": page_index,
                    "category": "PageText",
                },
            )
        )
    return parsed


def parse_with_langchain_unstructured(path: Path, strategy: str) -> list[ParsedElement]:
    # This is the parser family the project expects: UnstructuredPDFParser from
    # LangChain Community. The fallback below uses unstructured directly because
    # package versions differ a little between environments.
    from langchain_community.document_loaders.blob_loaders import Blob
    from langchain_community.document_loaders.parsers.pdf import UnstructuredPDFParser

    parser = UnstructuredPDFParser(mode="elements", strategy=strategy)
    blob = Blob.from_path(str(path))
    documents = parser.lazy_parse(blob)
    return [
        ParsedElement(
            text=(document.page_content or "").strip(),
            metadata=dict(document.metadata or {}),
        )
        for document in documents
        if (document.page_content or "").strip()
    ]


def parse_with_unstructured_direct(path: Path, strategy: str) -> list[ParsedElement]:
    from unstructured.partition.pdf import partition_pdf

    elements = partition_pdf(
        filename=str(path),
        strategy=strategy,
        infer_table_structure=True,
        include_page_breaks=False,
    )
    parsed: list[ParsedElement] = []
    for element in elements:
        text = str(element).strip()
        if not text:
            continue
        metadata = {}
        if getattr(element, "metadata", None):
            metadata = element.metadata.to_dict()
        metadata["category"] = element.__class__.__name__
        parsed.append(ParsedElement(text=text, metadata=metadata))
    return parsed


def parse_pdf(path: Path, strategy: str) -> tuple[list[ParsedElement], str]:
    if strategy == "pypdf":
        return parse_with_pypdf(path), "pypdf"

    try:
        return parse_with_langchain_unstructured(path, strategy), "langchain_unstructured_pdf_parser"
    except Exception as first_error:
        try:
            elements = parse_with_unstructured_direct(path, strategy)
            return elements, "unstructured_partition_pdf"
        except Exception as second_error:
            try:
                return parse_with_pypdf(path), "pypdf_fallback"
            except Exception as third_error:
                raise RuntimeError(
                    f"Cannot parse {path}. "
                    f"UnstructuredPDFParser error: {first_error}; "
                    f"partition_pdf error: {second_error}; "
                    f"pypdf fallback error: {third_error}"
                ) from third_error


def element_page(metadata: dict[str, Any]) -> int | None:
    value = metadata.get("page_number") or metadata.get("page")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_chunks(
    elements: Iterable[ParsedElement],
    base_metadata: dict[str, Any],
    chunk_size: int,
    overlap: int,
    parser_name: str,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    buffer: list[str] = []
    pages: list[int] = []
    categories: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        text = clean_text("\n".join(buffer))
        if not text:
            buffer.clear()
            pages.clear()
            categories.clear()
            return
        page_numbers = sorted(set(pages))
        metadata = {
            **base_metadata,
            "parser": parser_name,
            "chunk_index": len(chunks),
            "char_count": len(text),
            "page_start": page_numbers[0] if page_numbers else None,
            "page_end": page_numbers[-1] if page_numbers else None,
            "element_categories": sorted(set(categories)),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        chunks.append({"text": text, "metadata": metadata})

        if overlap > 0 and len(text) > overlap:
            tail = text[-overlap:]
            buffer.clear()
            buffer.append(tail)
            pages.clear()
            pages.extend(page_numbers[-1:] if page_numbers else [])
            categories.clear()
        else:
            buffer.clear()
            pages.clear()
            categories.clear()

    for element in elements:
        text = clean_text(element.text)
        if not text:
            continue
        page = element_page(element.metadata)
        if page is not None:
            pages.append(page)
        category = element.metadata.get("category") or element.metadata.get("type")
        if category:
            categories.append(str(category))

        current_length = sum(len(item) for item in buffer) + len(buffer)
        if buffer and current_length + len(text) > chunk_size:
            flush()
        buffer.append(text)

    flush()
    return chunks


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_export_group(group: str | None) -> str:
    if not group or group == "root_policy_documents":
        return "group_5_vietnam_policies"
    return group


def chunk_header(chunk: dict[str, Any]) -> str:
    metadata = chunk["metadata"]
    page_start = metadata.get("page_start")
    page_end = metadata.get("page_end")
    pages = ""
    if page_start and page_end:
        pages = f" | pages {page_start}-{page_end}"
    elif page_start:
        pages = f" | page {page_start}"
    return f"--- chunk {metadata.get('chunk_index', '')}{pages} ---"


def document_header(metadata: dict[str, Any]) -> str:
    fields = [
        ("document_id", metadata.get("document_id")),
        ("source", metadata.get("source")),
        ("year", metadata.get("year")),
        ("country", metadata.get("country")),
        ("source_type", metadata.get("source_type")),
        ("collection_role", metadata.get("collection_role")),
    ]
    meta_lines = [f"- {key}: {value}" for key, value in fields if value]
    title = metadata.get("title") or metadata.get("document_id") or "Untitled document"
    return "\n".join([f"## {title}", *meta_lines]).strip()


def write_group_text_exports(text_dir: Path, chunks: list[dict[str, Any]]) -> None:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for chunk in chunks:
        metadata = chunk["metadata"]
        group = normalize_export_group(metadata.get("group"))
        document_id = metadata.get("document_id") or "unknown_document"
        grouped.setdefault(group, {}).setdefault(document_id, []).append(chunk)

    ordered_groups = [
        *[group for group in GROUP_EXPORT_ORDER if group in grouped],
        *sorted(group for group in grouped if group not in GROUP_EXPORT_ORDER),
    ]

    for group in ordered_groups:
        documents = grouped[group]
        title = GROUP_EXPORT_TITLES.get(group, group)
        sections = [f"# {title}"]
        for document_id in sorted(documents):
            document_chunks = sorted(
                documents[document_id],
                key=lambda item: item["metadata"].get("chunk_index", 0),
            )
            sections.append(document_header(document_chunks[0]["metadata"]))
            for chunk in document_chunks:
                sections.append(f"{chunk_header(chunk)}\n{chunk['text']}")
        (text_dir / f"{group}.txt").write_text(
            "\n\n".join(sections).strip() + "\n",
            encoding="utf-8",
        )


def write_document_text_exports(text_dir: Path, chunks: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        document_id = chunk["metadata"]["document_id"]
        grouped.setdefault(document_id, []).append(chunk)

    for document_id, document_chunks in grouped.items():
        document_chunks = sorted(
            document_chunks,
            key=lambda item: item["metadata"].get("chunk_index", 0),
        )
        texts = [f"{chunk_header(chunk)}\n{chunk['text']}" for chunk in document_chunks]
        (text_dir / f"{document_id}.txt").write_text(
            "\n\n".join(texts),
            encoding="utf-8",
        )


def write_text_exports(
    text_dir: Path,
    chunks: list[dict[str, Any]],
    export_mode: str = "group",
) -> None:
    text_dir.mkdir(parents=True, exist_ok=True)
    for old_text_file in text_dir.glob("*.txt"):
        old_text_file.unlink()
    if export_mode == "document":
        write_document_text_exports(text_dir, chunks)
        return
    write_group_text_exports(text_dir, chunks)


def load_chunks_jsonl(path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    if not path.exists():
        return chunks
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def build_base_metadata(
    path: Path,
    manifest_metadata: dict[str, Any],
    inferred_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    inferred_metadata = inferred_metadata or {}
    merged = {**inferred_metadata, **manifest_metadata}
    document_id = merged.get("document_id") or safe_document_id(path)
    return {
        "document_id": document_id,
        "source": path.name,
        "source_path": str(path),
        "title": merged.get("title") or path.stem,
        "group": merged.get("group", "uncategorized"),
        "group_label": merged.get("group_label"),
        "collection_role": merged.get("collection_role"),
        "year": merged.get("year"),
        "country": merged.get("country"),
        "organization": merged.get("organization"),
        "source_type": merged.get("source_type", "pdf"),
        "link": merged.get("link"),
        "use_for": merged.get("use_for"),
        "manifest_status": merged.get("status")
        or merged.get("manifest_status", "not_in_manifest"),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Parse policy/report PDFs into RAG-ready chunks with metadata."
    )
    parser.add_argument(
        "--input-dir",
        default=os.getenv("PDF_INPUT_DIR", "/workspace/data/knowledge/raw"),
    )
    parser.add_argument(
        "--manifest",
        default=os.getenv("KNOWLEDGE_MANIFEST", "/workspace/configs/knowledge_manifest.yaml"),
    )
    parser.add_argument(
        "--output",
        default=os.getenv("PDF_CHUNKS_OUTPUT", "/workspace/data/knowledge/processed/chunks.jsonl"),
    )
    parser.add_argument(
        "--text-output-dir",
        default=os.getenv("PDF_TEXT_OUTPUT_DIR", "/workspace/data/knowledge/text"),
    )
    parser.add_argument(
        "--strategy",
        default=os.getenv("PDF_PARSE_STRATEGY", "pypdf"),
        choices=["pypdf", "fast", "hi_res", "ocr_only"],
    )
    parser.add_argument(
        "--text-export-mode",
        default=os.getenv("PDF_TEXT_EXPORT_MODE", "group"),
        choices=["group", "document"],
        help="group = write 5 grouped knowledge files; document = one text file per PDF.",
    )
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only rebuild text exports from an existing chunks.jsonl; do not parse PDFs.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list discovered PDFs and inferred metadata; do not parse.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Parse at most this many PDFs. 0 means all files.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    manifest = load_manifest(Path(args.manifest))
    output_path = Path(args.output)
    if args.export_only:
        chunks = load_chunks_jsonl(output_path)
        write_text_exports(
            Path(args.text_output_dir),
            chunks,
            export_mode=args.text_export_mode,
        )
        print(f"Exported {len(chunks)} existing chunks to {args.text_export_mode} text files.")
        print(f"Text output: {args.text_output_dir}")
        return

    pdf_paths = sorted(input_dir.rglob("*.pdf"))
    if args.max_files > 0:
        pdf_paths = pdf_paths[: args.max_files]
    if not pdf_paths:
        print(f"No PDFs found in {input_dir}. Add files and rerun this job.")
        write_jsonl(Path(args.output), [])
        return

    if args.list_only:
        rows = []
        for pdf_path in pdf_paths:
            manifest_metadata = find_manifest_metadata(pdf_path, manifest)
            inferred_metadata = infer_metadata_from_path(pdf_path, input_dir)
            metadata = build_base_metadata(
                pdf_path,
                manifest_metadata,
                inferred_metadata,
            )
            rows.append(
                {
                    "source": pdf_path.name,
                    "relative_path": str(pdf_path.relative_to(input_dir)),
                    "document_id": metadata["document_id"],
                    "group": metadata["group"],
                    "group_label": metadata.get("group_label"),
                    "source_type": metadata.get("source_type"),
                    "matched_manifest": bool(manifest_metadata),
                }
            )
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    all_chunks: list[dict[str, Any]] = []
    report: list[dict[str, Any]] = []
    for pdf_path in pdf_paths:
        manifest_metadata = find_manifest_metadata(pdf_path, manifest)
        inferred_metadata = infer_metadata_from_path(pdf_path, input_dir)
        base_metadata = build_base_metadata(
            pdf_path,
            manifest_metadata,
            inferred_metadata,
        )
        try:
            elements, parser_name = parse_pdf(pdf_path, args.strategy)
            chunks = build_chunks(
                elements=elements,
                base_metadata=base_metadata,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                parser_name=parser_name,
            )
            all_chunks.extend(chunks)
            report.append(
                {
                    "source": pdf_path.name,
                    "document_id": base_metadata["document_id"],
                    "status": "parsed",
                    "parser": parser_name,
                    "elements": len(elements),
                    "chunks": len(chunks),
                    "matched_manifest": bool(manifest_metadata),
                    "group": base_metadata.get("group"),
                    "group_label": base_metadata.get("group_label"),
                }
            )
        except Exception as exc:
            report.append(
                {
                    "source": pdf_path.name,
                    "document_id": base_metadata["document_id"],
                    "status": "error",
                    "error": str(exc),
                    "matched_manifest": bool(manifest_metadata),
                }
            )

    write_jsonl(output_path, all_chunks)
    write_text_exports(
        Path(args.text_output_dir),
        all_chunks,
        export_mode=args.text_export_mode,
    )
    report_path = output_path.with_name("parse_report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Parsed {len(pdf_paths)} PDFs into {len(all_chunks)} chunks.")
    print(f"Chunks: {output_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
