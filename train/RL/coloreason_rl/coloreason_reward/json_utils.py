import json
from typing import Any


def extract_json_object(text: str) -> dict[str, Any]:
    """Return the final top-level JSON object from a model completion."""
    decoder = json.JSONDecoder()
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, length = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append((index, index + length, value))
    if not candidates:
        raise ValueError("completion does not contain a JSON object")
    terminal = [candidate for candidate in candidates if not text[candidate[1] :].strip()]
    if terminal:
        return min(terminal, key=lambda candidate: candidate[0])[2]
    return max(candidates, key=lambda candidate: (candidate[1] - candidate[0], candidate[0]))[2]


def parse_reference(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("solution must be a JSON object or an encoded JSON object")


def canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
