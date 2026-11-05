"""Minimal Anthropic Messages API client (stdlib only) with forced tool output."""
import json
import os
import urllib.request

API_URL = "https://api.anthropic.com/v1/messages"
FAST_MODEL = os.environ.get("LEARNLOOP_FAST_MODEL", "claude-sonnet-5-5")  # grading, explanations
DEEP_MODEL = os.environ.get("LEARNLOOP_DEEP_MODEL", "claude-opus-5-5")    # card writing, scheduler tuning


def _api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Set ANTHROPIC_API_KEY to use learnloop.")
    return key


def _strict(schema):
    """Structured outputs want closed objects and no array-length constraints."""
    if isinstance(schema, dict):
        out = {k: _strict(v) for k, v in schema.items() if k not in ("minItems", "maxItems")}
        if out.get("type") == "object":
            out["additionalProperties"] = False
        return out
    if isinstance(schema, list):
        return [_strict(v) for v in schema]
    return schema


def structured(model: str, system: str, user: str, name: str, schema: dict, max_tokens: int = 4000,
               effort: str = "medium") -> dict:
    """Call the model with a JSON-schema output format; return the parsed object. `name` is for errors only."""
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": _strict(schema)}},
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode(),
        headers={
            "x-api-key": _api_key(),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Anthropic API {e.code}: {e.read().decode()[:500]}") from None
    text = "".join(b.get("text", "") for b in data["content"] if b["type"] == "text")
    if data.get("stop_reason") != "end_turn" or not text:
        raise RuntimeError(f"{name}: no complete output (stop_reason={data.get('stop_reason')})")
    return json.loads(text)
