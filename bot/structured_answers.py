from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServiceHit:
    clinic_name: str
    service_name: str
    price_rub: int | None
    patient_age_min_years: int | None
    duration_minutes: int | None


def _find_catalog_path(context_dir: str) -> Path | None:
    base = Path(context_dir)
    if not base.exists():
        return None
    # Exact name used in our test context.
    direct = base / "каталог_клиник.json"
    if direct.exists():
        return direct
    # Fallback: any json that ends with this name.
    for p in base.rglob("*каталог_клиник.json"):
        if p.is_file():
            return p
    return None


def _load_catalog(context_dir: str) -> dict | None:
    p = _find_catalog_path(context_dir)
    if p is None:
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(p.read_text(encoding="cp1251", errors="replace"))


def answer_eeg(context_dir: str) -> str | None:
    """
    Deterministic answer for EEG questions.
    Returns a fully formatted response (Answer/Основание) or None if no EEG data exists.
    """
    catalog = _load_catalog(context_dir)
    if not catalog:
        return None

    hits: list[ServiceHit] = []
    for cl in catalog.get("clinics", []):
        clinic_name = cl.get("name")
        for svc in cl.get("services", []):
            code = str(svc.get("code", ""))
            name = str(svc.get("name", ""))
            if not code.startswith("EEG_") and "ЭЭГ" not in name.upper():
                continue
            hits.append(
                ServiceHit(
                    clinic_name=str(clinic_name),
                    service_name=name,
                    price_rub=svc.get("price_rub"),
                    patient_age_min_years=svc.get("patient_age_min_years"),
                    duration_minutes=svc.get("duration_minutes"),
                )
            )

    if not hits:
        return None

    lines: list[str] = ["Ответ:"]
    for h in hits:
        parts: list[str] = [h.clinic_name, h.service_name]
        if h.patient_age_min_years is not None:
            parts.append(f"возраст min {h.patient_age_min_years} лет")
        if h.duration_minutes is not None:
            parts.append(f"длительность {h.duration_minutes} мин")
        if h.price_rub is not None:
            parts.append(f"цена {h.price_rub} руб.")
        lines.append(f"- " + " → ".join(parts))

    lines.append("")
    lines.append("Основание:")
    lines.append("- каталог_клиник.json")
    return "\n".join(lines)


def extract_eeg_facts(context_dir: str) -> str | None:
    """
    Extract a small, high-signal subset of catalog facts for EEG.
    This is meant to be *fed into the LLM* to prevent it from missing EEG entries in a large context.
    """
    catalog = _load_catalog(context_dir)
    if not catalog:
        return None

    facts: list[str] = []
    for cl in catalog.get("clinics", []):
        clinic_name = cl.get("name")
        for svc in cl.get("services", []):
            code = str(svc.get("code", ""))
            name = str(svc.get("name", ""))
            if not code.startswith("EEG_") and "ЭЭГ" not in name.upper():
                continue

            age = svc.get("patient_age_min_years")
            dur = svc.get("duration_minutes")
            price = svc.get("price_rub")

            facts.append(
                f"- {clinic_name} | {name} | code={code} | age_min_years={age} | duration_min={dur} | price_rub={price}"
            )

    if not facts:
        return None

    header = "EEG_FACTS (извлечено из каталог_клиник.json):"
    return header + "\n" + "\n".join(facts)

