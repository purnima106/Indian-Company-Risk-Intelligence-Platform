from app.llm.base import LLMClient

class GroqClient(LLMClient):
    def generate(self, prompt: str) -> str:
        raise NotImplementedError("GroqClient does not implement the generate method yet.")