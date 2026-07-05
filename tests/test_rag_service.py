import json

from apps.api.app.services.rag_service import RagService


def test_rag_service_uses_local_chunks_when_embeddings_disabled(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    rows = [
        {
            "text": "Vietnam should improve R&D investment and digital government capacity.",
            "metadata": {
                "document_id": "policy_a",
                "group": "group_5_vietnam_policies",
                "source": "policy_a.pdf",
            },
        },
        {
            "text": "This paragraph discusses unrelated tourism data.",
            "metadata": {"document_id": "other"},
        },
    ]
    chunks_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )

    service = RagService(
        qdrant_url="http://localhost:6333",
        collection="test",
        embedding_provider="disabled",
        embedding_model="",
        chunks_path=chunks_path,
    )

    results = service.search("R&D digital government Vietnam", limit=1)

    assert service.enabled is True
    assert len(results) == 1
    assert results[0].metadata["document_id"] == "policy_a"
