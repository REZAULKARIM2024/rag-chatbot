"""
usage.py
Cost, token and latency tracking for LLM calls.

Every call made through `call()` appends a record (label, tokens, latency,
estimated cost) to a list. `summarize()` turns the list into totals, so the
UI and tests can show what one question really cost.

PRICES are ILLUSTRATIVE defaults in USD per million tokens. Check the
current Anthropic pricing page and change them (or pass your own table)
before quoting any dollar figure.
"""

import time

PRICES = {  # model -> (input $/MTok, output $/MTok); illustrative, verify before use
    "default": (3.00, 15.00),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int, prices: dict | None = None) -> float:
    table = prices or PRICES
    in_price, out_price = table.get(model, table["default"])
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


def call(client, records: list, label: str, prices: dict | None = None, **kwargs):
    """Runs client.messages.create(**kwargs) and appends a usage record to `records`."""
    started = time.perf_counter()
    response = client.messages.create(**kwargs)
    elapsed = time.perf_counter() - started
    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    records.append({
        "label": label,
        "model": kwargs.get("model", ""),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_s": round(elapsed, 4),
        "cost_usd": estimate_cost(kwargs.get("model", ""), input_tokens, output_tokens, prices),
    })
    return response


def summarize(records: list) -> dict:
    return {
        "calls": len(records),
        "input_tokens": sum(r["input_tokens"] for r in records),
        "output_tokens": sum(r["output_tokens"] for r in records),
        "latency_s": round(sum(r["latency_s"] for r in records), 4),
        "cost_usd": sum(r["cost_usd"] for r in records),
    }
