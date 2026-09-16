import json
from urllib.error import URLError,HttpError
from urlib.request import urlopen, Request

from app.llm.base import LLMClient

class OllamaClient(LLMClient):
    def __init__(self, model: str = "llama3.2:3b", base_url: str = "http://localhost:11434", timeout: int = 120):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        payload = json.dump(
            {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            }

      ).encode("utf-8")

        request = Request(
            url=f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
            )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response.read().decode("utf-8"))
        except (URLError, HttpError, TimeoutError) as error:
            raise RuntimeError(f"Failed to connect to Ollama API: {error}") from error

        response_text = result.get("response")
        if not isinstance(response_text, str) or not response_text:
            raise ValueError("Invalid response from Ollama API: 'response' field is missing or not a string.")
        return response_text