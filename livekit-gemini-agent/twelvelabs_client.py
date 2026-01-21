import os
from dataclasses import dataclass
from typing import Any

import requests

try:
    from twelvelabs import TwelveLabs
except Exception:  # pragma: no cover - fallback handled at runtime
    TwelveLabs = None


DEFAULT_BASE_URL = "https://api.twelvelabs.io/v1.3"


@dataclass
class TwelveLabsResult:
    ok: bool
    data: dict[str, Any] | None = None
    error: str | None = None


class TwelveLabsClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("TWELVELABS_API_KEY", "").strip()
        self.base_url = os.getenv("TWELVELABS_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        self.index_id = os.getenv("TWELVELABS_INDEX_ID", "").strip()
        self._client = None
        if self.api_key and TwelveLabs is not None:
            self._client = TwelveLabs(api_key=self.api_key, base_url=self.base_url)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
        }

    def start_indexing(self, video_path: str, index_id: str | None = None) -> TwelveLabsResult:
        if not self.enabled:
            return TwelveLabsResult(ok=False, error="TWELVELABS_API_KEY is not set")

        index_id = index_id or self.index_id
        if not index_id:
            return TwelveLabsResult(ok=False, error="TWELVELABS_INDEX_ID is not set")

        if self._client is not None:
            try:
                task = self._client.tasks.create(index_id=index_id, video_file=video_path)
                payload = {
                    "task_id": getattr(task, "id", None),
                    "video_id": getattr(task, "video_id", None),
                }
                return TwelveLabsResult(ok=True, data=payload)
            except Exception as exc:
                return TwelveLabsResult(ok=False, error=str(exc))

        url = f"{self.base_url}/tasks"
        try:
            with open(video_path, "rb") as f:
                files = {"video_file": f}
                data = {"index_id": index_id}
                response = requests.post(
                    url,
                    headers=self._headers(),
                    data=data,
                    files=files,
                    timeout=120,
                )
            if response.status_code >= 400:
                return TwelveLabsResult(ok=False, error=f"Indexing failed: {response.text}")
            return TwelveLabsResult(ok=True, data=response.json())
        except Exception as exc:
            return TwelveLabsResult(ok=False, error=str(exc))

    def get_task(self, task_id: str) -> TwelveLabsResult:
        if not self.enabled:
            return TwelveLabsResult(ok=False, error="TWELVELABS_API_KEY is not set")

        if self._client is not None:
            try:
                task = self._client.tasks.retrieve(task_id)
                payload = {
                    "task_id": getattr(task, "id", None),
                    "video_id": getattr(task, "video_id", None),
                    "status": getattr(task, "status", None),
                }
                return TwelveLabsResult(ok=True, data=payload)
            except Exception as exc:
                return TwelveLabsResult(ok=False, error=str(exc))

        url = f"{self.base_url}/tasks/{task_id}"
        try:
            response = requests.get(url, headers=self._headers(), timeout=30)
            if response.status_code >= 400:
                return TwelveLabsResult(ok=False, error=f"Task fetch failed: {response.text}")
            return TwelveLabsResult(ok=True, data=response.json())
        except Exception as exc:
            return TwelveLabsResult(ok=False, error=str(exc))

    def generate_answer(self, video_id: str, question: str) -> TwelveLabsResult:
        if not self.enabled:
            return TwelveLabsResult(ok=False, error="TWELVELABS_API_KEY is not set")

        if self._client is not None:
            try:
                response = self._client.analyze(video_id=video_id, prompt=question)
                payload = {
                    "data": getattr(response, "data", None),
                }
                return TwelveLabsResult(ok=True, data=payload)
            except Exception as exc:
                return TwelveLabsResult(ok=False, error=str(exc))

        url = f"{self.base_url}/generate"
        payload = {
            "video_id": video_id,
            "type": "text",
            "prompt": question,
        }
        try:
            response = requests.post(
                url,
                headers={**self._headers(), "Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )
            if response.status_code >= 400:
                return TwelveLabsResult(ok=False, error=f"Generate failed: {response.text}")
            return TwelveLabsResult(ok=True, data=response.json())
        except Exception as exc:
            return TwelveLabsResult(ok=False, error=str(exc))


def extract_generated_text(payload: dict[str, Any] | None) -> str | None:
    if not payload:
        return None

    # Common API shapes: {"data": {"text": "..."}}, {"text": "..."}, {"summary": "..."}
    if isinstance(payload.get("text"), str):
        return payload.get("text")
    data = payload.get("data")
    if isinstance(data, dict) and isinstance(data.get("text"), str):
        return data.get("text")
    if isinstance(payload.get("summary"), str):
        return payload.get("summary")
    return None
