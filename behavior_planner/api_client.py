"""
OpenAI API client for the behavior planner.

This module keeps the system instruction in the model context only once per
program run. A root response is created once from the system instruction, and
all later prompt-only requests branch from that same root response id.
"""

from __future__ import annotations

import importlib
import os
import threading
from typing import Tuple


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DOTENV_PATH = os.path.join(PROJECT_ROOT, ".env")


def _load_dotenv_file(dotenv_path: str = DEFAULT_DOTENV_PATH) -> None:
    """Populate os.environ from a simple project-local .env file if present."""

    if not os.path.exists(dotenv_path):
        return

    with open(dotenv_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = str(raw_line).strip()
            if len(line) == 0 or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = str(key).strip()
            value = str(value).strip().strip('"').strip("'")
            if len(key) == 0:
                continue
            os.environ.setdefault(key, value)


class BehaviorPlannerAPIClient:
    """Small wrapper around the OpenAI Responses API for behavior planning."""

    def __init__(
        self,
        api_key_env_var: str = "OPENAI_API_KEY",
        model: str = "gpt-4o",
        temperature: float = 0.0,
        request_timeout_s: float = 30.0,
        max_output_tokens: int = 300,
        enabled: bool = True,
    ) -> None:
        self.api_key_env_var = str(api_key_env_var or "OPENAI_API_KEY").strip()
        self.model = str(model or "gpt-4o").strip()
        self.temperature = float(max(0.0, temperature))
        self.request_timeout_s = float(max(1.0, request_timeout_s))
        self.max_output_tokens = int(max(32, max_output_tokens))
        self.enabled = bool(enabled)
        self._client = None
        self._system_response_id: str | None = None
        self._system_response_lock = threading.Lock()
        self.api_key = self._resolve_api_key()

    def _resolve_api_key(self) -> str:
        _load_dotenv_file()
        return str(os.environ.get(self.api_key_env_var, "")).strip()

    def is_ready(self) -> bool:
        self.api_key = self._resolve_api_key()
        return bool(self.enabled and len(self.api_key) > 0)

    def _get_client(self):
        if self._client is not None:
            return self._client

        openai_module = importlib.import_module("openai")
        openai_client_cls = getattr(openai_module, "OpenAI")
        self._client = openai_client_cls(
            api_key=self.api_key,
            timeout=self.request_timeout_s,
        )
        return self._client

    @staticmethod
    def _extract_output_text(response) -> str:
        output_text = getattr(response, "output_text", "")
        if isinstance(output_text, str) and len(output_text.strip()) > 0:
            return output_text.strip()

        chunks = []
        for output_item in list(getattr(response, "output", []) or []):
            for content_item in list(getattr(output_item, "content", []) or []):
                text_value = getattr(content_item, "text", "")
                if isinstance(text_value, str) and len(text_value.strip()) > 0:
                    chunks.append(text_value.strip())
        return "\n".join(chunks).strip()

    def request_decision(
        self,
        system_instruction: str,
        prompt: str,
    ) -> Tuple[str, str | None]:
        """
        Request one behavior-planner decision.

        The system instruction is sent only once to create a root response.
        Every later prompt-only request references that same root response id so
        calls do not need to serialize through the previous prompt/response.
        """

        if not self.enabled:
            raise RuntimeError("Behavior-planner API client is disabled.")
        self.api_key = self._resolve_api_key()
        if len(self.api_key) == 0:
            raise RuntimeError(
                f"Behavior-planner API key is not configured. Set {self.api_key_env_var} in .env or the shell environment."
            )

        client = self._get_client()
        system_response_id = self._ensure_system_instruction_session(
            client=client,
            system_instruction=system_instruction,
        )
        request_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "previous_response_id": system_response_id,
        }
        input_payload = str(prompt)

        response = client.responses.create(
            input=input_payload,
            **request_kwargs,
        )
        response_id = getattr(response, "id", None)

        output_text = self._extract_output_text(response)
        if len(output_text) == 0:
            raise RuntimeError("Behavior-planner API returned an empty response.")
        return output_text, response_id

    def prime_system_instruction(self, system_instruction: str) -> str:
        """
        Send only the system instruction once and cache the root response id.

        This is used before the live prompt loop starts so the initial
        instruction latency does not block the first environment-input request.
        """

        if not self.enabled:
            raise RuntimeError("Behavior-planner API client is disabled.")
        self.api_key = self._resolve_api_key()
        if len(self.api_key) == 0:
            raise RuntimeError(
                f"Behavior-planner API key is not configured. Set {self.api_key_env_var} in .env or the shell environment."
            )
        client = self._get_client()
        return self._ensure_system_instruction_session(
            client=client,
            system_instruction=system_instruction,
        )

    def _ensure_system_instruction_session(self, client, system_instruction: str) -> str:
        with self._system_response_lock:
            if isinstance(self._system_response_id, str) and len(self._system_response_id) > 0:
                return self._system_response_id

            response = client.responses.create(
                model=self.model,
                temperature=self.temperature,
                max_output_tokens=32,
                input=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "input_text",
                                "text": str(system_instruction),
                            }
                        ],
                    }
                ],
            )
            response_id = getattr(response, "id", None)
            if not isinstance(response_id, str) or len(response_id) == 0:
                raise RuntimeError("Behavior-planner API failed to create the system-instruction session.")
            self._system_response_id = response_id
            return self._system_response_id
