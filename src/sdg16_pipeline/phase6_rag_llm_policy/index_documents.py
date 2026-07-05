import os
import uuid
import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.sdg16_pipeline.phase6_rag_llm_policy.embeddings import (
    EmbeddingClient,
    EmbeddingDisabledError,
)


def chunk_text(text: str, size: int = 1200, overlap: int = 150) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap
    return [chunk for chunk in chunks if chunk]


def load_jsonl_documents(path: Path) -> list[tuple[str, int, str, dict]]:
    documents = []
    if not path.exists():
        return documents
    with path.open("r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text", "").strip()
            if not text:
                continue
            metadata = row.get("metadata", {})
            source = metadata.get("source") or metadata.get("document_id") or path.name
            chunk_index = int(metadata.get("chunk_index", index))
            documents.append((source, chunk_index, text, metadata))
    return documents


def load_txt_documents(knowledge_dir: Path) -> list[tuple[str, int, str, dict]]:
    documents = []
    for path in knowledge_dir.rglob("*.txt"):
        if "processed" in path.parts:
            continue
        for index, chunk in enumerate(chunk_text(path.read_text(encoding="utf-8"))):
            documents.append(
                (
                    path.name,
                    index,
                    chunk,
                    {"source": path.name, "source_path": str(path), "chunk_index": index},
                )
            )
    return documents


def main() -> None:
    knowledge_dir = Path(os.getenv("KNOWLEDGE_DIR", "/workspace/data/knowledge"))
    chunks_path = Path(
        os.getenv(
            "RAG_CHUNKS_PATH",
            str(knowledge_dir / "processed" / "chunks.jsonl"),
        )
    )
    collection = os.getenv("QDRANT_COLLECTION", "sdg16_knowledge")
    embedder = EmbeddingClient(
        provider=os.getenv("EMBEDDING_PROVIDER", "disabled"),
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("LLM_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", ""),
        ollama_url=os.getenv("OLLAMA_URL", "http://ollama:11434"),
    )
    try:
        embedder.ensure_enabled()
    except EmbeddingDisabledError as exc:
        raise SystemExit(str(exc)) from exc

    documents = load_jsonl_documents(chunks_path)
    if not documents:
        documents = load_txt_documents(knowledge_dir)

    if not documents:
        raise SystemExit(
            f"No chunks found. Expected {chunks_path} or .txt documents in {knowledge_dir}."
        )

    first_vector = embedder.embed(documents[0][2])
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://qdrant:6333"))
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=len(first_vector),
                distance=Distance.COSINE,
            ),
        )

    points = []
    for position, (source, index, chunk, metadata) in enumerate(documents):
        vector = first_vector if position == 0 else embedder.embed(chunk)
        points.append(
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{index}")),
                vector=vector,
                payload={
                    "text": chunk,
                    "metadata": metadata,
                },
            )
        )
    client.upsert(collection_name=collection, points=points)
    print(f"Indexed {len(points)} chunks into {collection}.")


if __name__ == "__main__":
    main()
