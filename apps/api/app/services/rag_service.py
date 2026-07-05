import json
import math
import re
from pathlib import Path

try:
    from qdrant_client import QdrantClient
except ModuleNotFoundError:  # pragma: no cover - local lexical RAG can run without Qdrant.
    QdrantClient = None

from apps.api.app.schemas import SearchHit
from rag.embeddings import EmbeddingClient, EmbeddingDisabledError


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"\w+", text, flags=re.UNICODE)]


def compact_text(text: str, max_chars: int = 1400) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def metadata_text(metadata: dict) -> str:
    keys = [
        "document_id",
        "source",
        "title",
        "group",
        "group_label",
        "collection_role",
        "country",
        "source_type",
        "use_for",
    ]
    return " ".join(str(metadata.get(key, "")) for key in keys)


def policy_context_boost(query_set: set[str], metadata: dict) -> float:
    group = metadata.get("group")
    country = str(metadata.get("country", "")).lower()
    source_type = str(metadata.get("source_type", "")).lower()

    boost = 0.0
    if "vietnam" in query_set or "việt" in query_set or "nam" in query_set:
        if country == "vietnam":
            boost += 1.0
        if group in {"group_2_vietnam_country_reports", "group_5_vietnam_policies"}:
            boost += 0.8
        if country and country != "vietnam":
            boost -= 0.8
        if not country and group not in {
            "group_2_vietnam_country_reports",
            "group_5_vietnam_policies",
        }:
            boost -= 0.35
    if "fdi" in query_set and group in {
        "group_2_vietnam_country_reports",
        "group_5_vietnam_policies",
        "group_3_country_case_studies",
    }:
        boost += 0.2
    if {"policy", "chính", "sách"} & query_set and (
        group == "group_5_vietnam_policies" or "policy" in source_type
    ):
        boost += 0.3
    return boost


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
        chunks_path: Path | None = None,
        knowledge_dir: Path | None = None,
    ):
        self.collection = collection
        self.client = QdrantClient(url=qdrant_url) if QdrantClient else None
        self.chunks_path = chunks_path
        self.knowledge_dir = knowledge_dir
        self._local_documents: list[tuple[str, dict, set[str], dict[str, int]]] | None = None
        self.embedder = EmbeddingClient(
            provider=embedding_provider,
            model=embedding_model,
            api_key=api_key,
            base_url=base_url,
            ollama_url=ollama_url,
        )

    def search(self, query: str, limit: int = 5) -> list[SearchHit]:
        if self._embedding_enabled:
            try:
                return self._semantic_search(query, limit)
            except Exception:
                # Qdrant/embedding có thể chưa sẵn sàng trong giai đoạn demo.
                # Khi đó fallback sang local lexical search trên chunks.jsonl.
                pass
        return self._local_search(query, limit)

    def _semantic_search(self, query: str, limit: int) -> list[SearchHit]:
        if self.client is None:
            raise RuntimeError("qdrant-client is not installed.")
        vector = self.embedder.embed(query)
        results = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=limit,
            with_payload=True,
        ).points
        return [
            SearchHit(
                text=compact_text((point.payload or {}).get("text", "")),
                score=float(point.score),
                metadata=(point.payload or {}).get("metadata", {}),
            )
            for point in results
        ]

    def _local_search(self, query: str, limit: int) -> list[SearchHit]:
        documents = self._load_local_documents()
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        query_set = set(query_tokens)

        scored: list[tuple[float, str, dict]] = []
        for text, metadata, token_set, token_counts in documents:
            overlap = query_set & token_set
            if not overlap:
                continue
            tf_score = sum(token_counts.get(token, 0) for token in overlap)
            coverage = len(overlap) / len(query_set)
            length_penalty = 1 / math.sqrt(max(len(token_set), 1))
            score = (
                coverage
                + 0.1 * tf_score
                + length_penalty
                + policy_context_boost(query_set, metadata)
            )
            scored.append((score, text, metadata))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            SearchHit(text=compact_text(text), score=round(score, 4), metadata=metadata)
            for score, text, metadata in scored[:limit]
        ]

    def _load_local_documents(self) -> list[tuple[str, dict, set[str], dict[str, int]]]:
        if self._local_documents is not None:
            return self._local_documents

        rows: list[tuple[str, dict, set[str], dict[str, int]]] = []
        if self.chunks_path and self.chunks_path.exists():
            with self.chunks_path.open("r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    text = payload.get("text", "").strip()
                    if not text:
                        continue
                    tokens = tokenize(f"{metadata_text(payload.get('metadata', {}))} {text}")
                    counts: dict[str, int] = {}
                    for token in tokens:
                        counts[token] = counts.get(token, 0) + 1
                    rows.append((text, payload.get("metadata", {}), set(tokens), counts))

        if not rows and self.knowledge_dir and self.knowledge_dir.exists():
            for path in self.knowledge_dir.rglob("*.txt"):
                text = path.read_text(encoding="utf-8").strip()
                if not text:
                    continue
                metadata = {"source": path.name, "source_path": str(path)}
                tokens = tokenize(f"{metadata_text(metadata)} {text}")
                counts: dict[str, int] = {}
                for token in tokens:
                    counts[token] = counts.get(token, 0) + 1
                rows.append((text, metadata, set(tokens), counts))

        self._local_documents = rows
        return rows

    @property
    def enabled(self) -> bool:
        return self._embedding_enabled or bool(self._load_local_documents())

    @property
    def _embedding_enabled(self) -> bool:
        try:
            self.embedder.ensure_enabled()
            return True
        except EmbeddingDisabledError:
            return False
