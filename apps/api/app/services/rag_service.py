from qdrant_client import QdrantClient

from apps.api.app.schemas import SearchHit
from rag.embeddings import EmbeddingClient, EmbeddingDisabledError


class RagService:
    def __init__(
        self,
        qdrant_url: str,
        collection: str,
        embedding_provider: str,
        embedding_model: str,
        api_key: str = "",
        base_url: str = "",
        ollama_url: str = "",
    ):
        self.collection = collection
        self.client = QdrantClient(url=qdrant_url)
        self.embedder = EmbeddingClient(
            provider=embedding_provider,
            model=embedding_model,
            api_key=api_key,
            base_url=base_url,
            ollama_url=ollama_url,
        )

    def search(self, query: str, limit: int = 5) -> list[SearchHit]:
        vector = self.embedder.embed(query)
        results = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=limit,
            with_payload=True,
        ).points
        return [
            SearchHit(
                text=(point.payload or {}).get("text", ""),
                score=float(point.score),
                metadata=(point.payload or {}).get("metadata", {}),
            )
            for point in results
        ]

    @property
    def enabled(self) -> bool:
        try:
            self.embedder.ensure_enabled()
            return True
        except EmbeddingDisabledError:
            return False

