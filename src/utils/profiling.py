from __future__ import annotations

from contextlib import nullcontext
from contextvars import ContextVar
from threading import Lock


try:
    import nsight  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    nsight = None


_annotation_lock = Lock()
_annotation_counts: dict[str, int] = {}
_annotation_active: ContextVar[bool] = ContextVar("simbay_nsight_annotation_active", default=False)


class _AnnotatedContext:
    def __init__(self, name: str) -> None:
        self._name = name
        self._token = None
        self._context = None

    def __enter__(self):
        if _annotation_active.get():
            return None
        self._token = _annotation_active.set(True)
        self._context = nsight.annotate(self._name)
        return self._context.__enter__()

    def __exit__(self, exc_type, exc, tb):
        if self._context is None:
            return False
        try:
            return self._context.__exit__(exc_type, exc, tb)
        finally:
            if self._token is not None:
                _annotation_active.reset(self._token)


def annotate(name: str):
    if nsight is None:
        return nullcontext()
    with _annotation_lock:
        next_index = _annotation_counts.get(name, 0) + 1
        _annotation_counts[name] = next_index
    return _AnnotatedContext(f"{name}_{next_index}")
