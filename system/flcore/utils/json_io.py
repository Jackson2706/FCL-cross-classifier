"""Safe helpers for machine-readable experiment artifacts."""

import json
import math
import os
import tempfile


def _sanitize_non_finite(value):
    """Return ``(sanitized_value, changed)`` for nested JSON-compatible data."""
    if isinstance(value, float) and not math.isfinite(value):
        return None, True
    if isinstance(value, dict):
        changed = False
        result = {}
        for key, item in value.items():
            clean, item_changed = _sanitize_non_finite(item)
            result[key] = clean
            changed = changed or item_changed
        return result, changed
    if isinstance(value, list):
        result = []
        changed = False
        for item in value:
            clean, item_changed = _sanitize_non_finite(item)
            result.append(clean)
            changed = changed or item_changed
        return result, changed
    if isinstance(value, tuple):
        clean, changed = _sanitize_non_finite(list(value))
        return clean, changed
    return value, False


def atomic_write_json(path, payload, artifact_name=None):
    """Sanitize, serialize, and atomically replace a JSON artifact."""
    clean, sanitized = _sanitize_non_finite(payload)
    label = artifact_name or os.path.basename(os.fspath(path))
    if sanitized:
        print(f"[JSON] Warning: replaced non-finite values with null in {label}.")
    serialized = json.dumps(clean, indent=2, sort_keys=True, allow_nan=False) + "\n"
    directory = os.path.dirname(os.path.abspath(os.fspath(path)))
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=f".{os.path.basename(os.fspath(path))}.", dir=directory)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise

