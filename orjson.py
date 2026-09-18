"""Small stdlib fallback for environments where native orjson is policy-blocked."""

import json
from datetime import date, datetime
from typing import Any

JSONDecodeError = json.JSONDecodeError
OPT_PASSTHROUGH_DATETIME = 0
OPT_SERIALIZE_NUMPY = 0


def _default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def dumps(value: Any, *, option: int = 0, default=None, **kwargs: Any) -> bytes:
    serializer = default or _default
    return json.dumps(value, default=serializer, **kwargs).encode("utf-8")


def loads(value: Any, **kwargs: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value, **kwargs)
