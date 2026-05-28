import logging
import time
from dataclasses import dataclass
from typing import Protocol

from app.rag.prompts import SYSTEM_PROMPT
from app.utils.config import Settings, get_settings

logger = logging.getLogger(__name__)


class GenerationError(RuntimeError):
    """Raised when answer generation fails in a controlled way."""


class ChatClient(Protocol):
    def create(self, **kwargs): ...


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    model: str
    prompt_chars: int
    response_chars: int


class GroqGenerationService:
    """Reusable Groq-backed grounded answer generation service."""

    def __init__(
        self,
        settings: Settings | None = None,
        chat_completions: ChatClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.model = self.settings.generation_model
        self.timeout_seconds = self.settings.generation_timeout_seconds
        self._chat_completions = chat_completions

    @property
    def chat_completions(self) -> ChatClient:
        if self._chat_completions is None:
            if not self.settings.groq_api_key:
                raise GenerationError("GROQ_API_KEY is not configured.")
            try:
                from groq import Groq
            except ImportError as exc:
                raise GenerationError("groq package is required for answer generation.") from exc

            client = Groq(
                api_key=self.settings.groq_api_key,
                timeout=self.timeout_seconds,
            )
            self._chat_completions = client.chat.completions
        return self._chat_completions

    def generate(self, prompt: str) -> GenerationResult:
        prompt_chars = len(prompt)
        start = time.perf_counter()
        logger.info(
            "rag.generation_started",
            extra={"model": self.model, "prompt_chars": prompt_chars},
        )

        try:
            response = self.chat_completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=700,
            )
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.exception(
                "rag.generation_failed",
                extra={"model": self.model, "elapsed_ms": elapsed_ms},
            )
            raise GenerationError(
                "Answer generation failed. Please retry or check Groq configuration."
            ) from exc

        answer = response.choices[0].message.content.strip()
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "rag.generation_completed",
            extra={
                "model": self.model,
                "elapsed_ms": elapsed_ms,
                "response_chars": len(answer),
            },
        )
        return GenerationResult(
            answer=answer,
            model=self.model,
            prompt_chars=prompt_chars,
            response_chars=len(answer),
        )
