from __future__ import annotations

import json
import os
from pathlib import Path


def _read_text_file(path: Path) -> str:
    # Try utf-8 first; fallback to cp1251 (common for RU docs) if needed.
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="cp1251", errors="replace")


def load_context(context_dir: str, char_limit: int) -> str:
    base = Path(context_dir)
    if not base.exists() or not base.is_dir():
        return ""

    parts: list[str] = []
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue

        suffix = p.suffix.lower()
        rel = str(p.relative_to(base)).replace("\\", "/")

        try:
            if suffix in {".txt", ".md"}:
                content = _read_text_file(p).strip()
            elif suffix in {".json"}:
                raw = _read_text_file(p)
                obj = json.loads(raw)
                content = json.dumps(obj, ensure_ascii=False, indent=2).strip()
            else:
                continue
        except Exception:
            continue

        if not content:
            continue

        parts.append(f"===== SOURCE: {rel} =====\n{content}\n")

    ctx = "\n".join(parts).strip()
    if not ctx:
        return ""

    if len(ctx) > char_limit:
        ctx = ctx[:char_limit] + "\n\n[TRUNCATED: context_char_limit reached]"

    return ctx

