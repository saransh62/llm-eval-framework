"""On-disk cache for judge verdicts so re-running an eval never pays for the same judgement twice."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

DEFAULT_CACHE_DIR = Path(__file__).parent / ".cache" / "judge"


def cache_key(*parts: str) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()


class JudgeCache:
    def __init__(self, directory: Path = DEFAULT_CACHE_DIR, enabled: bool = True):
        self.directory = directory
        self.enabled = enabled
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        path = self._path(key)
        if self.enabled and path.exists():
            self.hits += 1
            return json.loads(path.read_text())
        self.misses += 1
        return None

    def put(self, key: str, value: dict) -> None:
        if not self.enabled:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False))
        tmp.replace(path)
