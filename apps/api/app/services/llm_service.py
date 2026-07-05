import httpx

try:
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover - Ollama/local disabled mode does not need it.
    OpenAI = None


class LlmService:
    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        base_url: str,
        ollama_url: str,
        ollama_model: str,
    ):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.ollama_url = ollama_url.rstrip("/")
        self.ollama_model = ollama_model

    @property
    def enabled(self) -> bool:
        return self.provider in {"openai", "openai-compatible", "ollama"}

    def answer(self, system_prompt: str, user_prompt: str) -> str:
        if self.provider in {"openai", "openai-compatible"}:
            if OpenAI is None:
                raise RuntimeError("Install openai to use OpenAI-compatible chat models.")
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url or None,
            )
            response = client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content or ""

        if self.provider == "ollama":
            response = httpx.post(
                f"{self.ollama_url}/api/chat",
                timeout=120,
                json={
                    "model": self.ollama_model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
            )
            response.raise_for_status()
            return response.json()["message"]["content"]

        return (
            "LLM đang tắt. Hãy đặt LLM_PROVIDER=openai, openai-compatible "
            "hoặc ollama để bật khuyến nghị sinh tự động."
        )
