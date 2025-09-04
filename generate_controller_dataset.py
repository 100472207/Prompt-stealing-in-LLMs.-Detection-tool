#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_controller_dataset.py
--------------------------------
Genera controller_dataset.jsonl adaptado a la arquitectura:
  - Input:  <<CONTEXT_ID=CX>>\nProblem: ...
  - Output: [CONTEXT_ID=CX] [ROUTE=SIMPLE|COMPLEX] Problem: ...

NOTAS:
- Registra conteos finales por categoría (SIMPLE / COMPLEX).
- Guarda controller_dataset_stats.json con:
    • total de ejemplos
    • distribución por categoría (conteo y porcentaje)
    • ejemplos representativos (hasta 5 por categoría)
"""

from __future__ import annotations
import argparse
import json
import logging
import multiprocessing as mp
import random
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

from gen_executors_datasets import download_mathematics_dataset, generate_examples

# ────────────────────────── logging ──────────────────────────
log = logging.getLogger("build-controller-ds")
logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")

# ─────────────── ID de contexto ↔ descripción ───────────────
FILTER_CONTEXTS: Dict[str, Tuple[str, str]] = {
    "arithmetic__add_or_sub":        ("C1", "Solve this addition or subtraction problem. Return only the result."),
    "arithmetic__add_sub_multiple":  ("C2", "Solve this multi-step arithmetic problem. Output the final answer."),
    "arithmetic__mixed":             ("C3", "Solve this mixed arithmetic problem step by step. Return only the result."),
    "arithmetic__mul":               ("C4", "You are solving a multiplication problem. Output the result only."),
    "arithmetic__mul_div_multiple":  ("C5", "Solve this multi-step multiplication/division problem. Output only the final answer."),
    "algebra__linear_1d":            ("C6", "Solve this linear equation in one variable. Return only the value of x."),
    "calculus__differentiate":       ("C7", "You are solving a calculus problem involving differentiation."),
    "polynomials__add":              ("C8", "Add the following polynomials and return the simplified result."),
    "polynomials__collect":          ("C9", "Collect like terms and simplify the polynomial expression."),
    "polynomials__compose":          ("C10", "Compose the polynomials and provide the final result."),
    "polynomials__evaluate":         ("C11", "Evaluate the polynomial expression at the given value."),
    "polynomials__expand":           ("C12", "Expand and simplify the polynomial expression."),
}

MODEL_FILTERS = {
    "simple": list(FILTER_CONTEXTS.keys())[:6],   # C1..C6
    "complex": list(FILTER_CONTEXTS.keys())[6:],  # C7..C12
}

# ─────────────── Utilidades de parsing ───────────────
ANSI_ALL   = re.compile(r"\x1b\[[0-9;]*m")
GREEN_TAG  = "\x1b[92m"
SECTION_RE = re.compile(r"\x1b\[1m(?:train|interpolate)/")
PROBLEM_RE = re.compile(r"Problem:\s*(.*)$")

def _strip_ansi(text: str) -> str:
    return ANSI_ALL.sub("", text)

def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def _extract_questions(txt: Path) -> List[str]:
    """Extrae enunciados de problema de los .txt generados por mathematics_dataset."""
    out: List[str] = []
    if not txt.exists():
        return out
    parts: List[str] = []
    with txt.open(encoding="utf-8") as fh:
        for raw in fh:
            if SECTION_RE.match(raw):
                continue
            if GREEN_TAG in raw:
                before = raw.split(GREEN_TAG, 1)[0]
                parts.append(before)
                q = _normalize_ws(_strip_ansi("".join(parts)))
                if q:
                    out.append(q)
                parts.clear()
            else:
                parts.append(raw)
    if parts:
        q = _normalize_ws(_strip_ansi("".join(parts)))
        if q:
            out.append(q)
    return out

def _worker_gen(repo: Path, flt: str, out_txt: Path):
    generate_examples(repo, [flt], out_file=out_txt)

def _ensure_txt(repo: Path, flt: str, timeout: int) -> Path:
    """Asegura un .txt por filtro (genera con timeout si no existe)."""
    txt = Path("data/mathematics_dataset") / f"{flt}.txt"
    if txt.exists():
        return txt
    proc = mp.Process(target=_worker_gen, args=(repo, flt, txt), daemon=True)
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        raise TimeoutError(f"Timeout ({timeout}s) generando '{flt}'")
    return txt

# ─────────────── Dataset builder ───────────────
def build_dataset(seed: int, target_per_filter: int, timeout: int) -> list[dict]:
    """Construye ejemplos etiquetados con CONTEXT_ID y ROUTE equilibrando SIMPLE/COMPLEX."""
    random.seed(seed)
    repo = download_mathematics_dataset()
    samples: List[Dict[str, Any]] = []

    for route in ("simple", "complex"):
        for flt in MODEL_FILTERS[route]:
            context_id, _ = FILTER_CONTEXTS[flt]
            txt = _ensure_txt(repo, flt, timeout)
            qs = _extract_questions(txt)
            if not qs:
                log.warning("Sin preguntas válidas en %s", flt)
                continue

            uniq = list(dict.fromkeys(qs))  # dedup preservando orden
            picked = (
                random.sample(uniq, target_per_filter)
                if len(uniq) >= target_per_filter
                else uniq + random.choices(uniq, k=target_per_filter - len(uniq))
            )

            for q in picked:
                samples.append({
                    "input":     f"<<CONTEXT_ID={context_id}>>\nProblem: {q}",
                    "output":    f"[CONTEXT_ID={context_id}] [ROUTE={route.upper()}] Problem: {q}",
                    "category":  route.upper(),
                    "context_id": context_id
                })

        count_route = sum(1 for x in samples if x["category"] == route.upper())
        log.info("%s → %d ejemplos acumulados", route.upper(), count_route)

    # ───────── Balanceo estricto por categoría ─────────
    simple = [x for x in samples if x["category"] == "SIMPLE"]
    complex_ = [x for x in samples if x["category"] == "COMPLEX"]
    n = min(len(simple), len(complex_))
    if n == 0:
        raise RuntimeError("No se pudieron generar ejemplos para balancear SIMPLE/COMPLEX.")
    balanced = random.sample(simple, n) + random.sample(complex_, n)
    random.shuffle(balanced)
    return balanced

# ─────────────── Estadísticas y muestreo ───────────────
def _extract_problem(text: str) -> str:
    """Extrae el enunciado tras 'Problem:' (toma la última ocurrencia en la cadena)."""
    m = PROBLEM_RE.search(text)
    return m.group(1).strip() if m else text.strip()

def compute_stats(examples: List[Dict[str, Any]], sample_k: int = 5) -> Dict[str, Any]:
    total = len(examples)
    by_cat = {"SIMPLE": [], "COMPLEX": []}
    for ex in examples:
        by_cat.setdefault(ex["category"], []).append(ex)

    def pack_examples(items: List[Dict[str, Any]], k: int):
        if not items:
            return []
        picked = random.sample(items, k=min(k, len(items)))
        out = []
        for e in picked:
            out.append({
                "context_id": e.get("context_id"),
                "route": e.get("category"),
                "problem": _extract_problem(e.get("output", "")),
            })
        return out

    stats = {
        "total_examples": total,
        "distribution": {
            cat: {
                "count": len(items),
                "percent": round(100.0 * len(items) / total, 2) if total else 0.0
            } for cat, items in by_cat.items()
        },
        "representative_examples": {
            cat: pack_examples(items, sample_k) for cat, items in by_cat.items()
        }
    }
    return stats

# ─────────────── Main ───────────────
def main() -> None:
    p = argparse.ArgumentParser("Genera dataset para el controlador (predicción de CONTEXT_ID + ROUTE)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-per-filter", type=int, default=80,
                   help="Número objetivo de problemas por filtro (antes de balanceo).")
    p.add_argument("--timeout", type=int, default=90)
    p.add_argument("--out-dir", type=Path, default=Path("./data/controller"))
    p.add_argument("--samples-per-category", type=int, default=5,
                   help="Número de ejemplos representativos por categoría en el JSON de estadísticas.")
    args = p.parse_args()

    try:
        examples = build_dataset(args.seed, args.target_per_filter, args.timeout)

        # Guardar dataset
        args.out_dir.mkdir(parents=True, exist_ok=True)
        ds_file = args.out_dir / "controller_dataset.jsonl"
        with ds_file.open("w", encoding="utf-8") as fh:
            for e in examples:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
        log.info("✅ Dataset creado: %s  (total=%d)", ds_file, len(examples))

        # Guardar estadísticas y muestreo para 6.2.1
        stats = compute_stats(examples, sample_k=args.samples_per_category)
        stats_file = args.out_dir / "controller_dataset_stats.json"
        stats_file.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

        # Log resumido
        dist = stats["distribution"]
        log.info("📊 Distribución: SIMPLE=%d (%.2f%%) | COMPLEX=%d (%.2f%%)",
                 dist.get("SIMPLE", {}).get("count", 0),  dist.get("SIMPLE", {}).get("percent", 0.0),
                 dist.get("COMPLEX", {}).get("count", 0), dist.get("COMPLEX", {}).get("percent", 0.0))
        log.info("📝 Estadísticas guardadas en: %s", stats_file)

    except Exception as exc:
        log.error("❌  Error: %s", exc)
        sys.exit(1)

if __name__ == "__main__":
    main()
