import os
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag.embeddings import EmbeddingClient, EmbeddingDisabledError


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


def main() -> None:
    knowledge_dir = Path(os.getenv("KNOWLEDGE_DIR", "/workspace/data/knowledge"))
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

    documents = []
    for path in knowledge_dir.rglob("*.txt"):
        for index, chunk in enumerate(chunk_text(path.read_text(encoding="utf-8"))):
            documents.append((path, index, chunk))

    if not documents:
        raise SystemExit(
            f"No .txt documents found in {knowledge_dir}. "
            "Parse PDFs with Unstructured before indexing."
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
    for position, (path, index, chunk) in enumerate(documents):
        vector = first_vector if position == 0 else embedder.embed(chunk)
        points.append(
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{path}:{index}")),
                vector=vector,
                payload={
                    "text": chunk,
                    "metadata": {"source": path.name, "chunk": index},
                },
            )
        )
    client.upsert(collection_name=collection, points=points)
    print(f"Indexed {len(points)} chunks into {collection}.")


if __name__ == "__main__":
    main()

