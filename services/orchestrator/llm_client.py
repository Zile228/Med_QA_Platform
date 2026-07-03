"""
services/orchestrator/llm_client.py
LLM client -- supports 6 backends: Ollama (local), Google Gemini (cloud),
OpenAI (cloud), RemoteInferenceClient (fine-tuned model on a rented pod),
LocalHFClient (fine-tuned model running locally, eval only), and Mock (dev/test).

Backend is chosen via env LLM_BACKEND='ollama' | 'google' | 'openai' | 'remote' | 'local_hf' | 'mock'.
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
    # Set True by subclasses with a real generate_with_image(). hasattr()
    # cannot be used since the default method exists on every subclass.
    _supports_multimodal: bool = False

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

    def generate_with_image(
        self,
        image_bytes: bytes,
        prompt: str,
        system: Optional[str] = None,
        mime_type: str = "image/png",
    ) -> str:
        """
        Multimodal call: send an image + text prompt, get back a text response.

        Not abstract on purpose, so OllamaClient/MockLLMClient are not forced
        to implement it. Since this default exists on every subclass, hasattr()
        cannot detect real support -- callers must check the
        _supports_multimodal flag instead.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support multimodal input."
        )


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

    _supports_multimodal = True

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

    def generate_with_image(
        self,
        image_bytes: bytes,
        prompt: str,
        system: Optional[str] = None,
        mime_type: str = "image/png",
    ) -> str:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        image_part = self._genai.types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type,
        )
        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=[image_part, full_prompt],
            )
            return response.text.strip()
        except Exception as e:
            raise RuntimeError(
                f"[GoogleGeminiClient] generate_with_image failed: {e}"
            )


# Client for OpenAI

class OpenAIClient(BaseLLMClient):
    """
    Calls the OpenAI Chat Completions API via the 'openai' SDK.
    Requires OPENAI_API_KEY in env.
    Default model: gpt-4o-mini (supports vision, usable for both CoT text and
    BI-RADS/TI-RADS image description).
    """

    _supports_multimodal = True

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai is not installed. Run: pip install openai")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in env.")

        self.model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._client = OpenAI(api_key=self.api_key)
        print(f"[llm] OpenAIClient -> model: {self.model_name}")

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
            content = resp.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            raise RuntimeError(f"[OpenAIClient] Generate failed: {e}")

    def chat(
        self,
        messages: List[dict],
        system: Optional[str] = None,
    ) -> str:
        full_messages = list(messages)
        if system:
            full_messages = [{"role": "system", "content": system}] + full_messages
        try:
            resp = self._client.chat.completions.create(
                model=self.model_name,
                messages=full_messages,
            )
            content = resp.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            raise RuntimeError(f"[OpenAIClient] Chat failed: {e}")

    def generate_with_image(
        self,
        image_bytes: bytes,
        prompt: str,
        system: Optional[str] = None,
        mime_type: str = "image/png",
    ) -> str:
        import base64

        b64_image = base64.b64encode(image_bytes).decode("ascii")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64_image}"},
                },
            ],
        })
        try:
            resp = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
            content = resp.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            raise RuntimeError(f"[OpenAIClient] generate_with_image failed: {e}")


# Client for a fine-tuned model deployed on a rented pod (RunPod, Vast.ai, Modal...)

class RemoteInferenceClient(BaseLLMClient):
    """
    Calls a fine-tuned Qwen model served by vLLM on a rented pod, via vLLM's
    OpenAI-compatible /v1/chat/completions endpoint. Falls back to
    GoogleGeminiClient automatically when the pod is unreachable or times out.

    vLLM is used instead of a plain transformers pipeline because it supports
    continuous batching -- several concurrent requests (e.g. during a live
    demo) get processed in parallel instead of queued one-by-one.

    Env vars:
        REMOTE_INFERENCE_URL   Base URL of the vLLM server on the pod, WITHOUT
                                /v1 suffix (e.g. https://xxxx.runpod.net or
                                http://<ip>:<port>). The client appends /v1/...
                                itself.
        REMOTE_MODEL_NAME      Model name vLLM was started with (the value
                                passed to "vllm serve <model>", or to
                                --served-model-name if set explicitly).
        REMOTE_INFERENCE_TOKEN Bearer token if the pod requires auth (matches
                                vLLM's --api-key; leave empty if not set).
        REMOTE_TIMEOUT         Timeout in seconds (default: 30).
        GOOGLE_API_KEY         Used by the fallback client.

    Used when LLM_BACKEND=remote.

    Note: generate_with_image() is not implemented -- the BI-RADS node still
    uses GoogleGeminiClient regardless of this backend.
    """

    def __init__(
        self,
        base_url: str = None,
        model_name: str = None,
        token: str = None,
        fallback_client: "BaseLLMClient" = None,
        timeout: int = None,
    ):
        import httpx

        self.base_url = (
            base_url or os.getenv("REMOTE_INFERENCE_URL", "")
        ).rstrip("/")
        self.model_name = model_name or os.getenv("REMOTE_MODEL_NAME", "")
        self.token = token or os.getenv("REMOTE_INFERENCE_TOKEN", "")
        self.timeout = timeout or int(os.getenv("REMOTE_TIMEOUT", "30"))
        # Once a pod call fails, stop retrying for the rest of this client's
        # lifetime instead of stalling every future request on the timeout.
        self._pod_alive = bool(self.base_url and self.model_name)
        if self.base_url and not self.model_name:
            print(
                "[llm] WARNING: REMOTE_INFERENCE_URL is set but REMOTE_MODEL_NAME "
                "is not -- vLLM requires a model name on every request. "
                "RemoteInferenceClient will use the fallback client until "
                "REMOTE_MODEL_NAME is set."
            )

        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        self._client = httpx.Client(timeout=self.timeout, headers=headers)

        self._fallback = fallback_client or GoogleGeminiClient()
        print(
            f"[llm] RemoteInferenceClient -> {self.base_url or '(no URL set)'} "
            f"(model={self.model_name or '(not set)'}) "
            f"| fallback: {type(self._fallback).__name__}"
        )

    def _chat_completion(self, messages: List[dict]) -> str:
        resp = self._client.post(
            f"{self.base_url}/v1/chat/completions",
            json={"model": self.model_name, "messages": messages},
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return content.strip() if content else ""

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        if self._pod_alive:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            try:
                return self._chat_completion(messages)
            except Exception as e:
                print(
                    f"[RemoteInferenceClient] Pod unreachable: {e} "
                    f"-- falling back to {type(self._fallback).__name__}"
                )
                self._pod_alive = False

        return self._fallback.generate(prompt, system)

    def chat(
        self,
        messages: List[dict],
        system: Optional[str] = None,
    ) -> str:
        if self._pod_alive:
            full_messages = list(messages)
            if system:
                full_messages = [{"role": "system", "content": system}] + full_messages
            try:
                return self._chat_completion(full_messages)
            except Exception as e:
                print(
                    f"[RemoteInferenceClient] Pod unreachable: {e} "
                    f"-- falling back to {type(self._fallback).__name__}"
                )
                self._pod_alive = False

        return self._fallback.chat(messages, system)


# Client for a fine-tuned model running locally (CPU, eval only)

class LocalHFClient(BaseLLMClient):
    """
    Runs a locally fine-tuned HuggingFace model.
    Used when LLM_BACKEND=local_hf and HF_MODEL_PATH points to the model directory.
    Inference is slow on CPU -- intended for eval only, not production/demo.

    Note: generate_with_image() is not implemented -- the BI-RADS node still
    needs GoogleGeminiClient regardless of this backend.
    """

    def __init__(self, model_path: str = None):
        from transformers import pipeline as hf_pipeline, AutoTokenizer

        self.model_path = model_path or os.getenv("HF_MODEL_PATH", "")
        if not self.model_path:
            raise ValueError("HF_MODEL_PATH not set in env")

        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._pipe = hf_pipeline(
            "text-generation",
            model=self.model_path,
            tokenizer=tokenizer,
            max_new_tokens=512,
            device_map="auto",
        )
        print(f"[llm] LocalHFClient -> {self.model_path}")

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        result = self._pipe(messages)
        return result[0]["generated_text"][-1]["content"].strip()


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

def get_llm_client(backend_override: str = None) -> BaseLLMClient:
    """
    Reads LLM_BACKEND from env -> returns the matching client.
    Pass backend_override to use a different backend without changing env
    (e.g. GEVAL_LLM_BACKEND in eval_qa.py to avoid self-preference bias).

    ollama   -> OllamaClient (default)
    google   -> GoogleGeminiClient
    openai   -> OpenAIClient
    remote   -> RemoteInferenceClient (fine-tuned model on a rented pod)
    local_hf -> LocalHFClient (fine-tuned model running locally, eval only)
    mock     -> MockLLMClient
    """
    backend = (backend_override or os.getenv("LLM_BACKEND", "ollama")).lower()

    if backend == "ollama":
        return OllamaClient()
    elif backend == "google":
        return GoogleGeminiClient()
    elif backend == "openai":
        return OpenAIClient()
    elif backend == "remote":
        return RemoteInferenceClient()
    elif backend == "local_hf":
        return LocalHFClient()
    elif backend == "mock":
        return MockLLMClient()
    else:
        print(f"[llm] WARNING: LLM_BACKEND='{backend}' not recognized -> falling back to MockLLMClient")
        return MockLLMClient()
