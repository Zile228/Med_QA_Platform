"""
services/orchestrator/llm_client.py
=====================================
LLM client -- supports 2 backends: Ollama (local) and Google Gemini (cloud).

Backend is chosen via env LLM_BACKEND='ollama' | 'google'.
Swap backend = change 1 env var, no code changes needed.

Public API:
    get_llm_client() -> BaseLLMClient
    BaseLLMClient.generate(prompt: str, system: str) -> str
    BaseLLMClient.chat(messages: list[dict], system: str) -> str
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Optional, List


# Base class for LLM clients

class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Single-turn: send a prompt -> return the text response."""
        ...

    def chat(
        self,
        messages: List[dict],
        system: Optional[str] = None,
    ) -> str:
        """
        Multi-turn: takes a list of messages (role/content) -> returns a reply.

        The default implementation concatenates history into one prompt then
        calls generate(). Subclasses override this to use a native multi-turn
        API where available.

        messages: [{"role": "user"|"assistant", "content": str}, ...]
        """
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        )
        combined = f"{history_text}" if history_text else ""
        return self.generate(combined, system=system)


# Client for Ollama (local)

class OllamaClient(BaseLLMClient):
    """
    Calls the Ollama REST API at localhost:11434.
    No API key needed -- runs entirely locally.
    """

    def __init__(
        self,
        base_url: str = None,
        model: str = None,
        timeout: int = 120,
    ):
        import httpx
        self.base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "phi4-mini")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        print(f"[llm] OllamaClient -> {self.base_url} | model: {self.model}")

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        try:
            resp = self._client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            raise RuntimeError(f"[OllamaClient] Generate failed: {e}")

    def chat(
        self,
        messages: List[dict],
        system: Optional[str] = None,
    ) -> str:
        """Uses Ollama's /api/chat to keep native conversation context."""
        ollama_messages = []
        if system:
            ollama_messages.append({"role": "system", "content": system})
        ollama_messages.extend(messages)

        payload = {"model": self.model, "messages": ollama_messages, "stream": False}
        try:
            resp = self._client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "").strip()
        except Exception as e:
            raise RuntimeError(f"[OllamaClient] Chat failed: {e}")


# Client for Google Gemini

class GoogleGeminiClient(BaseLLMClient):
    """
    Calls the Google Gemini API via the 'google-genai' SDK.
    Requires GOOGLE_API_KEY in env.
    Default model: gemini-2.5-flash.
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
    ):
        try:
            from google import genai
        except ImportError:
            raise ImportError("google-genai is not installed. Run: pip install google-genai")

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in env.")

        self.model_name = model or os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
        self._client = genai.Client(api_key=self.api_key)
        self._genai = genai
        print(f"[llm] GoogleGeminiClient -> model: {self.model_name}")

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
            )
            return response.text.strip()
        except Exception as e:
            raise RuntimeError(f"[GoogleGeminiClient] Generate failed: {e}")

    def chat(
        self,
        messages: List[dict],
        system: Optional[str] = None,
    ) -> str:
        """
        Multi-turn using the google-genai contents list.
        The system prompt is prepended to the first user message.
        """
        contents = []
        for i, m in enumerate(messages):
            role = m.get("role", "user")
            content = m.get("content", "")
            if i == 0 and system and role == "user":
                content = f"{system}\n\n{content}"
            gemini_role = "model" if role == "assistant" else "user"
            contents.append({"role": gemini_role, "parts": [{"text": content}]})

        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=contents,
            )
            return response.text.strip()
        except Exception as e:
            raise RuntimeError(f"[GoogleGeminiClient] Chat failed: {e}")


# Mock client for dev/test

class MockLLMClient(BaseLLMClient):
    """
    Returns fixed template text.
    Used when LLM_BACKEND='mock' or no Ollama/Google key is available.
    """

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        return (
            "[MOCK LLM] This is a placeholder response. "
            "Configure LLM_BACKEND=ollama or LLM_BACKEND=google in .env to enable real LLM output."
        )

    def chat(
        self,
        messages: List[dict],
        system: Optional[str] = None,
    ) -> str:
        last = messages[-1].get("content", "") if messages else ""
        return f"[MOCK LLM] Echo: {last}"


# Factory: chooses the client based on env

def get_llm_client() -> BaseLLMClient:
    """
    Reads LLM_BACKEND from env -> returns the matching client.

    ollama  -> OllamaClient (default)
    google  -> GoogleGeminiClient
    mock    -> MockLLMClient
    """
    backend = os.getenv("LLM_BACKEND", "ollama").lower()

    if backend == "ollama":
        return OllamaClient()
    elif backend == "google":
        return GoogleGeminiClient()
    elif backend == "mock":
        return MockLLMClient()
    else:
        print(f"[llm] WARNING: LLM_BACKEND='{backend}' not recognized -> falling back to MockLLMClient")
        return MockLLMClient()
