"""
safety.py
Defences against prompt injection coming from retrieved documents.

In RAG, the documents are *untrusted input*: anyone who can get text into
the knowledge base can try to smuggle instructions to the model. This module
adds three deterministic layers that need no LLM call:

  1. scan_for_injection  - heuristic detector for instruction-like text.
  2. filter_results      - quarantines retrieved chunks the detector flags.
  3. wrap_context        - puts the remaining text inside <documents> tags
                           and neutralises attempts to close those tags, so
                           the model can be told "everything in here is data".

Limits (be honest in interviews): pattern matching catches common, blunt
attacks only. A paraphrased or non-English injection can slip through, so
this is defence in depth next to the hardened system prompt, not a guarantee.
eval/run_redteam.py measures what it does and does not catch.
"""

import re

# (name, regex) pairs. Names show up in reports and traces.
INJECTION_PATTERNS = [
    ("ignore-previous", r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b(previous|prior|above|earlier|preceding|system|all)\b[^.\n]{0,30}\b(instructions?|prompts?|rules|context|messages?)\b"),
    ("reveal-prompt", r"\b(reveal|print|show|repeat|output|leak|display)\b[^.\n]{0,30}\b(system prompt|hidden instructions?|your instructions|instructions above|initial prompt)\b"),
    ("role-reassign", r"\byou are now\b|\bfrom now on,? you\b|\bact as (an? )?(unrestricted|jailbroken|different|evil)"),
    ("new-instructions", r"\bnew (instructions?|task|rules?)\s*:"),
    ("fake-role-line", r"(?im)^\s*(system|assistant|developer)\s*:"),
    ("concealment", r"\bdo not (tell|inform|mention|reveal)\b[^.\n]{0,40}\b(user|anyone|human)\b"),
    ("exfiltration", r"\b(send|post|upload|email|forward|exfiltrate|leak)\b[^.\n]{0,60}(https?://|api[ _-]?key|secret|credentials)"),
    ("tag-breakout", r"</?\s*(documents?|system|instructions?|context)\s*>"),
    ("jailbreak-term", r"\bjailbreak\b|\bDAN mode\b|\bdeveloper mode\b"),
]
_COMPILED = [(name, re.compile(pattern, re.IGNORECASE)) for name, pattern in INJECTION_PATTERNS]

_TAG_RE = re.compile(r"</?\s*documents?\s*>", re.IGNORECASE)


def scan_for_injection(text: str) -> list[str]:
    """Returns the names of every injection pattern found in `text` (empty = looks clean)."""
    return [name for name, regex in _COMPILED if regex.search(text or "")]


def filter_results(results: list[dict]) -> tuple[list[dict], list[dict]]:
    """Splits retrieved chunks into (safe, quarantined). Quarantined ones carry a 'flags' list."""
    safe, quarantined = [], []
    for r in results:
        flags = scan_for_injection(r.get("text", ""))
        if flags:
            quarantined.append({**r, "flags": flags})
        else:
            safe.append(r)
    return safe, quarantined


def neutralise_tags(text: str) -> str:
    """Stops document text from closing or opening our <documents> wrapper."""
    return _TAG_RE.sub("[tag removed]", text)


def wrap_context(body: str) -> str:
    """Marks retrieved text as untrusted data for the model."""
    return f"<documents>\n{neutralise_tags(body)}\n</documents>"


def leaked_canary(answer: str, canary: str) -> bool:
    """True if a secret planted in the prompt (a canary) shows up in the answer."""
    return bool(canary) and canary.lower() in (answer or "").lower()
