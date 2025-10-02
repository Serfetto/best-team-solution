"""State management for the parsing service."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Dict


DEFAULT_STATE: Dict[str, Any] = {
    "banki": {
        "full": {"next_page": 1, "has_more": True},
        "latest_date": None,
        "latest_date_ids": [],
        "running": False,
    },
    "sravni": {
        "full": {"next_page": 0, "has_more": True},
        "latest_date": None,
        "latest_date_ids": [],
        "running": False,
    },
}


class StateStore:
    """Simple JSON-backed state store safe for per-process concurrency."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = RLock()

    def load(self) -> Dict[str, Any]:
        with self._lock:
            if not self._path.exists():
                return deepcopy(DEFAULT_STATE)
            raw = self._path.read_text("utf-8")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                backup = self._path.with_suffix(self._path.suffix + ".bak")
                backup.write_text(raw, "utf-8")
                data = {}
            return self._merge_with_default(data)

    def save(self, state: Dict[str, Any]) -> None:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temp = self._path.with_suffix(self._path.suffix + ".tmp")
            temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), "utf-8")
            temp.replace(self._path)

    def update(self, mutate_fn: Callable[[Dict[str, Any]], None]) -> Dict[str, Any]:
        with self._lock:
            state = self.load()
            mutate_fn(state)
            self.save(state)
            return state

    def _merge_with_default(self, data: Dict[str, Any]) -> Dict[str, Any]:
        merged = deepcopy(DEFAULT_STATE)
        for key, value in data.items():
            default_value = merged.get(key)
            if isinstance(default_value, dict) and isinstance(value, dict):
                merged[key] = self._merge_nested(default_value, value)
            else:
                merged[key] = value
        return merged

    def _merge_nested(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        result = deepcopy(base)
        for key, value in override.items():
            default_value = result.get(key)
            if isinstance(default_value, dict) and isinstance(value, dict):
                result[key] = self._merge_nested(default_value, value)
            else:
                result[key] = value
        return result
