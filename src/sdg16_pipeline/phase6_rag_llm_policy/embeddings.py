import httpx

try:
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover - not needed for local/Ollama demo.
    OpenAI = None


class EmbeddingDisabledError(RuntimeError):
    pass


class EmbeddingClient:
    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str = "",
        base_url: str = "",
        ollama_url: str = "",
    ):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.ollama_url = ollama_url.rstrip("/")

    def ensure_enabled(self) -> None:
        if self.provider not in {"openai", "openai-compatible", "ollama"}:
            raise EmbeddingDisabledError(
                "Set EMBEDDING_PROVIDER to openai, openai-compatible or ollama."
            )

    def embed(self, text: str) -> list[float]:
        self.ensure_enabled()
        if self.provider in {"openai", "openai-compatible"}:
            if OpenAI is None:
                raise RuntimeError("Install openai to use OpenAI-compatible embeddings.")
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url or None,
            )
            return client.embeddings.create(
                model=self.model,
                input=text,
            ).data[0].embedding

        response = httpx.post(
            f"{self.ollama_url}/api/embed",
            timeout=120,
            json={"model": self.model, "input": text},
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]
