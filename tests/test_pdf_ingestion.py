from pathlib import Path

from rag.parse_pdfs import (
    ParsedElement,
    build_chunks,
    infer_metadata_from_path,
    write_text_exports,
)


def test_build_chunks_preserves_policy_metadata():
    chunks = build_chunks(
        elements=[
            ParsedElement(
                text="Vietnam should improve digital public services.",
                metadata={"page_number": 3, "category": "NarrativeText"},
            )
        ],
        base_metadata={
            "document_id": "vietnam_digital_policy",
            "source": "policy.pdf",
            "title": "Digital Policy",
            "group": "group_5_vietnam_policies",
            "year": 2020,
            "country": "Vietnam",
            "organization": "Vietnam Government",
            "source_type": "policy_document",
            "link": "example",
            "use_for": "policy recommendation",
            "manifest_status": "has_link",
        },
        chunk_size=1200,
        overlap=150,
        parser_name="test_parser",
    )

    assert len(chunks) == 1
    assert chunks[0]["metadata"]["document_id"] == "vietnam_digital_policy"
    assert chunks[0]["metadata"]["page_start"] == 3
    assert chunks[0]["metadata"]["element_categories"] == ["NarrativeText"]


def test_infer_group_metadata_from_local_folder():
    input_dir = Path("tailieuLLM")
    path = input_dir / "Nhóm 4" / "Paper_Giai_thich_Mo_hinh_SHAP.pdf"

    metadata = infer_metadata_from_path(path, input_dir)

    assert metadata["group"] == "group_4_scientific_papers"
    assert metadata["group_label"] == "Nhóm 4"
    assert metadata["source_type"] == "scientific_paper"


def test_root_policy_documents_are_group_5():
    input_dir = Path("tailieuLLM")
    path = input_dir / "Luat_DauTu_2020.pdf"

    metadata = infer_metadata_from_path(path, input_dir)

    assert metadata["group"] == "group_5_vietnam_policies"
    assert metadata["group_label"] == "Nhóm 5"
    assert metadata["source_type"] == "policy_document"
    assert metadata["country"] == "Vietnam"


def test_write_text_exports_groups_by_knowledge_group(tmp_path):
    chunks = [
        {
            "text": "IMD methodology text",
            "metadata": {
                "document_id": "imd_wcy",
                "source": "imd.pdf",
                "title": "IMD WCY",
                "group": "group_1_imd_wef_reports",
                "source_type": "competitiveness_report",
                "chunk_index": 0,
                "page_start": 1,
                "page_end": 1,
            },
        },
        {
            "text": "Vietnam policy text",
            "metadata": {
                "document_id": "root_policy",
                "source": "policy.pdf",
                "title": "Vietnam Policy",
                "group": "root_policy_documents",
                "source_type": "policy_document",
                "chunk_index": 0,
                "page_start": 2,
                "page_end": 3,
            },
        },
    ]

    write_text_exports(tmp_path, chunks)

    exported = sorted(path.name for path in tmp_path.glob("*.txt"))
    assert exported == [
        "group_1_imd_wef_reports.txt",
        "group_5_vietnam_policies.txt",
    ]
    assert "IMD methodology text" in (tmp_path / "group_1_imd_wef_reports.txt").read_text(
        encoding="utf-8"
    )
    assert "Vietnam policy text" in (tmp_path / "group_5_vietnam_policies.txt").read_text(
        encoding="utf-8"
    )
