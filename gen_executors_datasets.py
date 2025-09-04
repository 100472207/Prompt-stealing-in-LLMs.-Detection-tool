#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gen_executors_datasets_multistage.py — extracción Q/A robusta (DeepMind), saneado y auditoría EM-friendly

NOVEDAD: curriculum SIMPLE multi-etapa. Permite generar ejemplos combinando
los 3 niveles (1, 2 y 3) en un solo dataset para el ejecutor "simple".

Objetivo: maximizar EM del ejecutor "simple" y dar señal estable al modelo.
Cambios clave respecto a versiones anteriores:

- Atención correcta: se escribe attention_mask REAL (1 para tokens no-pad, 0 para pad).
- Canon numérica estable (simple): TODA respuesta numérica (entero/decimal/fracción)
  se normaliza a entero o fracción irreducible con denominador > 0.
- Filtro de complejidad para "simple" (por defecto ON): elimina enunciados con
  demasiados paréntesis/operadores para estabilizar el aprendizaje inicial.
- Saneado conservador de preguntas; nunca recortar por '.' (solo por '?').
- Curriculum SIMPLE multi-etapa: seleccionar una o varias etapas (1,2,3).
- max_length_simple=256 (menos truncamiento del prefijo).

Correcciones añadidas en esta versión:
- Parcheo automático de incompatibilidades con NumPy 2.0 en el repo DeepMind
  (reemplazo de ndarray.itemset(...) por asignación indexada).
- Filtro defensivo: no volcar líneas de Traceback/errores al .txt si el generador fallara.
- Limpieza del .txt contaminado sin pares válidos antes del reintento.
"""

from __future__ import annotations
import os, re, json, random, subprocess, sys, time, io
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from num2words import num2words
from fractions import Fraction
from decimal import Decimal, InvalidOperation

# ───────────────────────── Config ─────────────────────────
SEED = int(os.getenv("SEED", 42))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
EXECUTORS_DIR = (DATA_DIR / "executors"); EXECUTORS_DIR.mkdir(parents=True, exist_ok=True)

# ───────────────────────── Filtros por ejecutor ─────────────────────────
SIMPLE_CURRICULUM: Dict[int, List[str]] = {
    1: ["arithmetic__add_or_sub", "arithmetic__mul"],
    2: ["arithmetic__add_or_sub", "arithmetic__mul", "arithmetic__mul_div_multiple", "arithmetic__mixed"],
    3: ["arithmetic__add_or_sub", "arithmetic__mul", "arithmetic__mul_div_multiple", "arithmetic__mixed", "algebra__linear_1d"],
}

COMPLEX_FILTERS: List[str] = [
    "polynomials__add",
    "polynomials__collect",
    "polynomials__compose",
    "polynomials__evaluate",
    "polynomials__expand",
]

# Contextos
CONTEXT_MAP: Dict[str, str] = {
    "arithmetic__add_or_sub":       "Solve this addition or subtraction problem. Return only the result.",
    "arithmetic__add_sub_multiple": "Solve this multi-step arithmetic problem. Output the final answer.",
    "arithmetic__mixed":            "Solve this mixed arithmetic problem. Return only the result.",
    "arithmetic__mul":              "You are solving a multiplication problem. Output the result only.",
    "arithmetic__mul_div_multiple": "Solve this multi-step multiplication/division problem. Output only the final answer.",
    "algebra__linear_1d":           "Solve this linear equation in one variable. Return only the value of x.",
    "calculus__differentiate":      "You are solving a calculus problem involving differentiation.",
    "polynomials__add":             "Add the following polynomials and return the simplified result.",
    "polynomials__collect":         "Collect like terms and simplify the polynomial expression.",
    "polynomials__compose":         "Compose the polynomials and provide the final result.",
    "polynomials__evaluate":        "Evaluate the polynomial expression at the given value.",
    "polynomials__expand":          "Expand and simplify the polynomial expression.",
}

# ───────────────────────── Regex/util ─────────────────────────
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
BOLD_HEADER_RE = re.compile(r"^\x1b\[1m.*?\x1b\[0m\s*$")  # p.ej., \x1b[1mtrain/arithmetic__add_or_sub\x1b[0m
SPACE_RE = re.compile(r"\s+")
FRACTION_RE = re.compile(r"^\s*([-+]?\d+)\s*/\s*([-+]?\d+)\s*$")
DECIMAL_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")

# ANSI answer patterns
ANSI_ANS_ONLY_RE = re.compile(r"^\s*\x1b\[92m(?P<ans>.*?)\x1b\[0m\s*$")
ANSI_QA_SAME_RE  = re.compile(r"^(?P<q>.*?)\s+\x1b\[92m(?P<ans>.*?)\x1b\[0m\s*$")

# Fallback conservador (sin color): Q ... <numero>
PLAIN_QA_SAME_RE = re.compile(
    r"^(?P<q>.*?[?.]|[-+/*\d\s\.]+?)\s+(?P<ans>[-+]?\d+(?:/\d+)?(?:\.\d+)?)\s*$"
)

# Detección de línea "pregunta razonable"
QUESTIONISH_RE = re.compile(
    r"(?i)\b(Calculate|Evaluate|Solve|Work out|What|Put together|Add|Sum|Subtract|Total of|In base|Distance)\b|[?]$|^[\s\-\+*/\d\.]+$"
)

TAIL_NUMERIC_RE = re.compile(r"\?\s+[-+0-9./\s]+$")
OTHER_BASE_RE = re.compile(r"\bIn\s+base\b", flags=re.IGNORECASE)
MULTISPACE_RE = re.compile(r"\s{2,}")
STUB_ONLY_RE = re.compile(r"^(calculate|evaluate|solve)\s*\.?$", flags=re.IGNORECASE)

# Líneas de error típicas a filtrar del generador (defensa)
TRACE_OR_ERROR_RE = re.compile(
    r"^(Traceback \(most recent call last\):|  File \".*\", line \d+,|[A-Za-z]+Error: )"
)

# ───────────────────── helpers OS ─────────────────────
def _run(cmd: str) -> None:
    proc = subprocess.run(cmd, shell=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Comando fallido: {cmd}")

def _patch_numpy20_incompatibilities(repo_dir: Path) -> None:
    """
    Parchea incompatibilidades conocidas con NumPy 2.0 en el repo mathematics_dataset:
      - ndarray.itemset(i, v) → arr[i] = v
      - (opcional) np.asscalar(x) → np.asarray(x).item()
    """
    # 1) polynomials.sample: reemplazar itemset(...)
    poly_sample = repo_dir / "mathematics_dataset" / "sample" / "polynomials.py"
    if poly_sample.exists():
        txt = poly_sample.read_text(encoding="utf-8")
        # Reemplazo genérico: <arr>.itemset(<idx>, <val>) → <arr>[<idx>] = <val>
        txt2 = re.sub(
            r"(\b[\w\.]+)\.itemset\(\s*([^,]+)\s*,\s*([^)]+)\s*\)",
            r"\1[\2] = \3",
            txt
        )
        if txt2 != txt:
            poly_sample.write_text(txt2, encoding="utf-8")

    # 2) (defensivo) reemplazar np.asscalar(...) si aparece en cualquier módulo
    for py in (repo_dir / "mathematics_dataset").rglob("*.py"):
        src = py.read_text(encoding="utf-8")
        src2 = re.sub(r"\bnp\.asscalar\((.*?)\)", r"np.asarray(\1).item()", src)
        if src2 != src:
            py.write_text(src2, encoding="utf-8")

def download_mathematics_dataset(
    repo_url: str = "https://github.com/deepmind/mathematics_dataset.git",
    local_dir: Path | str = DATA_DIR / "mathematics_dataset",
) -> Path:
    local_dir = Path(local_dir).resolve()
    if not local_dir.exists():
        _run(f"git clone {repo_url} {local_dir}")
        # Parchear ANTES de instalar en editable
        _patch_numpy20_incompatibilities(local_dir)
        _run(f"pip install -q -e {local_dir}")
    else:
        # Si ya existe, asegurar parche y (opcional) reinstalar editable
        _patch_numpy20_incompatibilities(local_dir)
    return local_dir

# ────────────────── Limpieza / canon ──────────────────
def _strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s or "")

def _normalize_fraction(ans: str) -> str:
    m = FRACTION_RE.match(ans or "")
    if not m:
        return (ans or "").strip()
    num, den = int(m.group(1)), int(m.group(2))
    if den == 0:
        return (ans or "").strip()
    from math import gcd
    g = gcd(num, den); num //= g; den //= g
    if den < 0:
        den = -den; num = -num
    return f"{num}/{den}" if den != 1 else str(num)

def _canon_answer_numeric(ans: str) -> Optional[str]:
    s = _strip_ansi(ans).strip()
    s = re.sub(r"^\s*[A-Za-z]\s*=\s*", "", s)
    s = re.sub(r"^\s*:+\s*", "", s)
    s = re.sub(SPACE_RE, " ", s).strip()

    if FRACTION_RE.match(s):
        return _normalize_fraction(s)

    if DECIMAL_RE.match(s):
        try:
            f = Fraction(Decimal(s))
            return f"{f.numerator}/{f.denominator}" if f.denominator != 1 else str(f.numerator)
        except (InvalidOperation, ValueError):
            return None

    return None

def _canon_answer(ans: str, *, allow_symbolic: bool) -> str:
    numeric = _canon_answer_numeric(ans)
    if numeric is not None:
        return numeric

    s = _strip_ansi(ans).strip()
    s = re.sub(r"^\s*[A-Za-z]\s*=\s*", "", s)
    s = re.sub(r"^\s*:+\s*", "", s)
    s = re.sub(SPACE_RE, " ", s).strip()
    return s if allow_symbolic else s

def _answer_is_plausible(ans: str, *, algebra_ok: bool) -> bool:
    if not ans:
        return False
    if algebra_ok:
        return bool(re.match(r"^[\s0-9A-Za-z_+\-*/^().=]+$", ans))
    else:
        return bool(re.match(r"^[\s0-9/\-]+$", ans))

# ────────────────── Saneado de preguntas ──────────────────
def sanitize_question(raw: str, *, strict: bool, min_chars: int, drop_other_bases: bool) -> Tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {"cut_at_qmark": False, "tail_removed": False, "stub": False, "other_base": False}
    q = _strip_ansi(raw or "").strip()
    if not q:
        return "", {**meta, "reason": "empty"}

    if drop_other_bases and OTHER_BASE_RE.search(q):
        return "", {**meta, "other_base": True, "reason": "other_base"}

    if "?" in q:
        q = q.split("?", 1)[0].strip() + "?"
        meta["cut_at_qmark"] = True

    if TAIL_NUMERIC_RE.search(q):
        q = TAIL_NUMERIC_RE.sub("?", q)
        meta["tail_removed"] = True

    q = MULTISPACE_RE.sub(" ", q).strip()

    if STUB_ONLY_RE.match(q):
        return "", {**meta, "stub": True, "reason": "stub_question"}

    if strict:
        has_digit = bool(re.search(r"\d", q))
        looks_linear = bool(re.search(r"\b(x|y|a|b|t)\b.*=", q))
        if not (has_digit or looks_linear):
            return "", {**meta, "reason": "no_digit_no_linear"}

    if len(q) < max(8, min_chars):
        return "", {**meta, "reason": "too_short"}

    return q, meta

# ────────────────── Parser Q/A desde líneas ──────────────────
def _split_q_a_from_line(raw_line: str, pending_q: Optional[str]) -> Tuple[Optional[Tuple[str,str]], Optional[str]]:
    if not raw_line:
        return None, pending_q

    if BOLD_HEADER_RE.match(raw_line):
        return None, None  # al ver un header, invalida pending_q

    line = raw_line.rstrip("\n")

    m = ANSI_QA_SAME_RE.match(line)
    if m:
        q = _strip_ansi(m.group("q")).strip()
        a = m.group("ans").strip()
        return (q, a), None

    m2 = ANSI_ANS_ONLY_RE.match(line)
    if m2 and pending_q:
        a = m2.group("ans").strip()
        q = _strip_ansi(pending_q).strip()
        return (q, a), None

    plain = _strip_ansi(line).strip()
    if plain and QUESTIONISH_RE.search(plain):
        return None, plain

    mp = PLAIN_QA_SAME_RE.match(plain)
    if mp:
        q = mp.group("q").strip(); a = mp.group("ans").strip()
        return (q, a), None

    return None, pending_q

def read_pairs_from_generator_txt(txt_file: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if not txt_file.exists():
        return pairs

    pending_q: Optional[str] = None
    with open(txt_file, encoding="utf-8") as fh:
        for raw in fh:
            item, pending_q = _split_q_a_from_line(raw, pending_q)
            if item:
                q, a = item
                if q and a:
                    if "?" in q:
                        q = q.split("?", 1)[0].strip() + "?"
                    pairs.append((q, a))

    seen = set(); uniq: List[Tuple[str, str]] = []
    for q, a in pairs:
        key = (q, a)
        if key in seen: continue
        seen.add(key); uniq.append((q, a))
    return uniq

# ─────────── Generación por streaming con conteo real ───────────
def stream_generate_min_pairs(repo_dir: Path, filt: str, out_file: Path, want_pairs: int, timeout_sec: int = 180) -> int:
    gen_py = Path(repo_dir) / "mathematics_dataset" / "generate.py"
    if not gen_py.exists():
        raise FileNotFoundError(f"generate.py no encontrado en {gen_py}")
    cmd = [sys.executable, str(gen_py), "--filter", filt]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    added_pairs = 0
    start = time.time()
    pending_q: Optional[str] = None
    with open(out_file, "a", encoding="utf-8") as fh:
        try:
            while True:
                if proc.stdout is None:
                    break
                line = proc.stdout.readline()
                if not line:
                    break

                # ⚠️ No volcar líneas de error/traceback al .txt
                if not TRACE_OR_ERROR_RE.match(line):
                    fh.write(line)

                item, pending_q = _split_q_a_from_line(line, pending_q)
                if item:
                    added_pairs += 1
                    if added_pairs >= want_pairs:
                        break

                if time.time() - start > timeout_sec:
                    break
        finally:
            try:
                proc.terminate(); proc.wait(timeout=2)
            except Exception:
                try: proc.kill()
                except Exception: pass
    return added_pairs

def ensure_min_pairs_in_txt(repo_dir: Path, filt: str, txt_file: Path, min_pairs: int) -> None:
    attempts = 0
    while attempts < 6:
        curr = read_pairs_from_generator_txt(txt_file)
        missing = max(0, min_pairs - len(curr))
        if missing <= 0:
            return

        # Si está contaminado y no tiene pares, limpiar el fichero antes de reintentar
        if attempts == 0 and len(curr) == 0 and txt_file.exists():
            try:
                txt_file.unlink()
            except Exception:
                pass

        got = stream_generate_min_pairs(repo_dir, filt, txt_file, want_pairs=missing, timeout_sec=180)
        attempts += 1
        # Si no añadió nada y sigue sin pares, cortar para no buclear
        if got == 0 and len(curr) == len(read_pairs_from_generator_txt(txt_file)):
            break

# ────────────────── Augment seguro/controlado ──────────────────
NUM_RE = re.compile(r"\b-?\d{1,3}(?:,\d{3})*(?:\.\d+)?\b")

def numbers_to_words(text: str) -> str:
    digit_map = {str(i): w for i, w in enumerate(
        ["zero","one","two","three","four","five","six","seven","eight","nine"]
    )}
    def _num_repl(m: re.Match) -> str:
        token = m.group()
        clean = token.replace(",", "")
        neg = clean.startswith("-")
        if neg: clean = clean[1:]
        if "." in clean:
            i, f = clean.split(".", 1)
            try: i_w = num2words(int(i), lang="en").replace(" and ", " ")
            except Exception: i_w = i
            f_w = " ".join(digit_map.get(d, d) for d in f)
            out = f"{i_w} point {f_w}"
        else:
            try: out = num2words(int(clean), lang="en").replace(" and ", " ")
            except Exception: return token
        return f"minus {out}" if neg else out
    processed = NUM_RE.sub(_num_repl, text)
    processed = re.sub(r"(?<![A-Za-z0-9])-(?=[A-Za-z])", "minus ", processed)
    return processed

def make_variants(prompt: str, n_variants: int, include_num2words: bool) -> List[str]:
    base = prompt.strip()
    variants = {base}
    if include_num2words and n_variants > 0:
        variants.add(numbers_to_words(base))
    # Conmutación A op B ↔ B op A para + y *
    pat = re.compile(r"\b(\d+)\s*([+*])\s*(\d+)\b")
    for m in pat.finditer(base):
        a, op, b = m.groups()
        variants.add(base.replace(m.group(), f"{b} {op} {a}"))
    lst = list(variants)
    random.shuffle(lst)
    return lst[: min(len(lst), max(1, n_variants + 1))]

# ────────────────── Codificación (attention_mask correcto) ──────────────────
def encode_example(
    context: str, question: str, answer: str,
    tokenizer: PreTrainedTokenizerBase, max_length: int
) -> Tuple[List[int], List[int], List[int]]:
    prefix = f"{context}\nProblem: {question}\nAnswer"
    pre = tokenizer(prefix, add_special_tokens=False).input_ids
    ans = tokenizer(answer, add_special_tokens=False).input_ids

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    nonpad = len(pre) + len(ans) + 1  # + eos
    if nonpad > max_length:
        cut = nonpad - max_length
        if cut >= len(pre):
            pre = pre[-1:]
        else:
            pre = pre[:-cut]
        nonpad = len(pre) + len(ans) + 1

    input_ids = pre + ans + [eos_id] + [pad_id] * (max_length - nonpad)
    labels    = [-100]*len(pre) + ans + [eos_id] + [-100]*(max_length - nonpad)
    attn_mask = [1]*nonpad + [0]*(max_length - nonpad)
    return input_ids, attn_mask, labels

# ────────────────── Leakage & complejidad ──────────────────
def _canon_simple(s: str) -> str:
    s = _strip_ansi(s or "").strip()
    s = re.sub(SPACE_RE, " ", s).strip()
    return s

def _answer_in_question(q: str, a: str) -> bool:
    q_c = _canon_simple(q)
    a_c = _canon_simple(_canon_answer(a, allow_symbolic=True))
    if not q_c or not a_c:
        return False
    if FRACTION_RE.match(a_c):
        num, den = a_c.split("/", 1)
        patterns = [a_c, f"{num} / {den}", f" {a_c} ", f" {num} / {den} "]
        return any(pat in q_c for pat in patterns)
    return a_c in q_c

_SIMPLE_PAREN_RE = re.compile(r"[()]")

def _too_complex_for_simple(q: str) -> bool:
    expr = q
    if _SIMPLE_PAREN_RE.search(expr):
        return True
    if expr.count('/') > 1:
        return True
    return False

# ────────────────── Estadísticas / Auditoría ──────────────────
def _init_stats(exec_name: str, filters: List[str], reps_per_filter: int) -> Dict[str, Any]:
    return {
        "executor": exec_name,
        "filters": {
            f: {"count": 0, "percent": 0.0, "context": CONTEXT_MAP.get(f, ""), "representative": []}
            for f in filters
        },
        "totals": {"train": 0, "test": 0, "overall": 0},
        "_reps_per_filter": reps_per_filter,
        "_discarded": {"count": 0, "reasons": {}},
        "_mask_tokens_avg": 0.0,
    }

def _note_discard(stats_exec: Dict[str, Any], reason: str):
    stats_exec["_discarded"]["count"] += 1
    stats_exec["_discarded"]["reasons"][reason] = stats_exec["_discarded"]["reasons"].get(reason, 0) + 1

def _maybe_add_rep(bucket: Dict[str, Any], ctx: str, q: str, a: str):
    reps = bucket["representative"]
    limit = bucket.get("_reps_limit", 3) or 3
    if len(reps) < limit:
        reps.append({"context": ctx, "problem": q, "answer": a})

def _finalize_stats(stats_exec: Dict[str, Any]) -> None:
    overall = stats_exec["totals"]["overall"]
    for _, b in stats_exec["filters"].items():
        c = b["count"]
        b["percent"] = round(100.0 * c / overall, 2) if overall else 0.0
    for k in list(stats_exec.keys()):
        if k.startswith("_") and k != "_discarded":
            del stats_exec[k]

# ────────────────── Construcción por ejecutor ──────────────────
def build_and_save(
    name: str,
    filters: List[str],
    tokenizer: PreTrainedTokenizerBase,
    target_per_filter: int,
    augment_variants: int,
    include_num2words: bool,
    max_length: int,
    split_ratio: float,
    reps_per_filter: int = 3,
    exclude_other_bases_for_simple: bool = True,
    strict_simple: bool = True,
    min_q_chars_simple: int = 12,
    strict_complex: bool = False,
    min_q_chars_complex: int = 8,
    write_audit_samples: int = 120,
    enable_simple_complexity_filter: bool = True,
) -> Dict[str, Any]:
    repo = download_mathematics_dataset()
    inputs, masks, labels = [], [], []
    stats = _init_stats(name, filters, reps_per_filter)

    audit: Dict[str, Any] = {
        "executor": name,
        "kept_samples": [],
        "discarded_examples": [],
        "discard_reasons": {},
        "filters": filters,
    }

    for f in filters:
        stats["filters"][f]["_reps_limit"] = reps_per_filter

    for f in filters:
        ctx = CONTEXT_MAP.get(f, "")
        txt_file = DATA_DIR / f"{f}.txt"

        ensure_min_pairs_in_txt(repo, f, txt_file, min_pairs=max(target_per_filter, 256))

        pairs = read_pairs_from_generator_txt(txt_file)
        if not pairs:
            _note_discard(stats, f"{f}:no_pairs")
            audit["discard_reasons"][f"{f}:no_pairs"] = audit["discard_reasons"].get(f"{f}:no_pairs", 0) + 1
            continue

        if len(pairs) >= target_per_filter:
            pairs = random.sample(pairs, target_per_filter)
        else:
            pairs = pairs + random.choices(pairs, k=target_per_filter - len(pairs))

        for (q_raw, a_raw) in pairs:
            strict = strict_simple if name == "simple" else strict_complex
            min_chars = min_q_chars_simple if name == "simple" else min_q_chars_complex

            q_clean, meta_q = sanitize_question(
                q_raw, strict=strict, min_chars=min_chars,
                drop_other_bases=(exclude_other_bases_for_simple if name == "simple" else False),
            )
            if not q_clean:
                reason = meta_q.get("reason", "invalid_question")
                _note_discard(stats, f"{f}:{reason}")
                audit["discarded_examples"].append({"filter": f, "q_raw": q_raw, "a_raw": a_raw, "meta": meta_q})
                audit["discard_reasons"][f"{f}:{reason}"] = audit["discard_reasons"].get(f"{f}:{reason}", 0) + 1
                continue

            if name == "simple" and enable_simple_complexity_filter and _too_complex_for_simple(q_clean):
                _note_discard(stats, f"{f}:too_complex")
                audit["discarded_examples"].append({"filter": f, "q": q_clean, "a_raw": a_raw, "meta": {**meta_q, "too_complex": True}})
                audit["discard_reasons"][f"{f}:too_complex"] = audit["discard_reasons"].get(f"{f}:too_complex", 0) + 1
                continue

            a = _canon_answer(a_raw, allow_symbolic=(name == "complex"))
            algebra_ok = (name == "complex")
            if not _answer_is_plausible(a, algebra_ok=algebra_ok):
                _note_discard(stats, f"{f}:implausible_answer")
                audit["discarded_examples"].append({"filter": f, "q_raw": q_raw, "q": q_clean, "a_raw": a_raw, "a": a, "meta": meta_q})
                audit["discard_reasons"][f"{f}:implausible_answer"] = audit["discard_reasons"].get(f"{f}:implausible_answer", 0) + 1
                continue

            if _answer_in_question(q_clean, a):
                _note_discard(stats, f"{f}:leakage_q_contains_a")
                audit["discarded_examples"].append({"filter": f, "q_raw": q_raw, "q": q_clean, "a_raw": a_raw, "a": a, "meta": {**meta_q, "leakage": True}})
                audit["discard_reasons"][f"{f}:leakage_q_contains_a"] = audit["discard_reasons"].get(f"{f}:leakage_q_contains_a", 0) + 1
                continue

            if stats["filters"][f]["count"] == 0:
                _maybe_add_rep(stats["filters"][f], ctx, q_clean, a)

            for qv in make_variants(q_clean, augment_variants, include_num2words=include_num2words):
                in_ids, attn, lbls = encode_example(ctx, qv, a, tokenizer, max_length)
                inputs.append(in_ids); masks.append(attn); labels.append(lbls)
                stats["filters"][f]["count"] += 1
                _maybe_add_rep(stats["filters"][f], ctx, qv, a)

            if len(audit["kept_samples"]) < write_audit_samples:
                audit["kept_samples"].append({"filter": f, "context": ctx, "problem": q_clean, "answer": a})

    ds = Dataset.from_dict({
        "input_ids": inputs, "attention_mask": masks, "labels": labels
    }).train_test_split(test_size=1 - split_ratio, seed=SEED)

    def _avg_labeled_tokens(split_ds: Dataset) -> float:
        tot = 0
        for i in range(len(split_ds)):
            y = split_ds[i]["labels"]
            tot += int(np.sum(np.array(y) != -100))
        return tot / max(1, len(split_ds))

    mask_avg_train = _avg_labeled_tokens(ds["train"])
    mask_avg_test  = _avg_labeled_tokens(ds["test"])
    stats["_mask_tokens_avg"] = round(float((mask_avg_train + mask_avg_test) / 2.0), 2)

    for split in ("train", "test"):
        out = EXECUTORS_DIR / f"dataset_{name}_{split}.json"
        rec = ds[split].to_dict()
        data = [{k: rec[k][i] for k in rec} for i in range(len(rec["input_ids"]))]
        with open(out, "w", encoding="utf-8") as fw:
            json.dump(data, fw, ensure_ascii=False, indent=2)
        stats["totals"][split] = len(data)
        print(f"✅ Guardado {name} {split}: {out}")

    stats["totals"]["overall"] = stats["totals"]["train"] + stats["totals"]["test"]
    _finalize_stats(stats)

    audit_out = EXECUTORS_DIR / f"executors_dataset_audit_{name}.json"
    with open(audit_out, "w", encoding="utf-8") as fa:
        json.dump(audit, fa, ensure_ascii=False, indent=2)
    print(f"🔎 Auditoría guardada en: {audit_out}")

    return stats

# ───────────────────────────── CLI ─────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("Genera datasets JSON para ejecutores simple/complex con parsing Q/A robusto (DeepMind) — EM-friendly, multi-etapa")
    ap.add_argument("--hf_token", type=str, required=True, help="Token de Hugging Face")
    ap.add_argument("--base_model", type=str, default="meta-llama/Llama-2-7b-chat-hf")

    # Tamaños por ejecutor (recomendado subir simple para EM alto)
    ap.add_argument("--target_per_filter_simple", type=int, default=200, help="Objetivo por subtipo (simple)")
    ap.add_argument("--target_per_filter_complex", type=int, default=180, help="Objetivo por subtipo (complex)")

    # Augment controlado
    ap.add_argument("--augment_variants_simple", type=int, default=0, help="Variantes por item (simple)")
    ap.add_argument("--augment_variants_complex", type=int, default=1, help="Variantes por item (complex)")

    # Longitudes
    ap.add_argument("--max_length_simple", type=int, default=256)
    ap.add_argument("--max_length_complex", type=int, default=256)

    # Split y reps
    ap.add_argument("--split_ratio", type=float, default=0.9)
    ap.add_argument("--reps_per_filter", type=int, default=3)

    # 🌟 Curriculum SIMPLE multi-etapa
    ap.add_argument(
        "--curriculum_simple_stages",
        type=str,
        default="1,2,3",
        help="Etapas del curriculum simple a incluir, separadas por coma (ej. '1', '1,2' o '1,2,3')."
    )

    # Filtros de calidad / saneado
    ap.add_argument("--allow_other_bases_in_simple", action="store_true",
                    help="Permite enunciados 'In base ...' en simple (por defecto se excluyen)")
    ap.add_argument("--no_strict_simple", action="store_true", help="Desactiva saneado estricto en simple")
    ap.add_argument("--min_q_chars_simple", type=int, default=12, help="Longitud mínima de pregunta (simple)")
    ap.add_argument("--strict_complex", action="store_true", help="Activa saneado también en complex")
    ap.add_argument("--min_q_chars_complex", type=int, default=8)
    ap.add_argument("--no_simple_complexity_filter", action="store_true",
                    help="Desactiva el filtro de complejidad para 'simple'")

    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, token=args.hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    all_stats: Dict[str, Any] = {"executors": []}

    try:
        selected_stages = [int(s.strip()) for s in args.curriculum_simple_stages.split(',') if s.strip()]
        for s in selected_stages:
            if s not in (1, 2, 3):
                raise ValueError
    except Exception:
        raise SystemExit("--curriculum_simple_stages debe ser una lista de etapas en {1,2,3}, por ejemplo '1,2,3' o '1'.")

    seen: set[str] = set()
    simple_filters: List[str] = []
    for stage in selected_stages:
        for f in SIMPLE_CURRICULUM[stage]:
            if f not in seen:
                seen.add(f)
                simple_filters.append(f)

    print(f"🔧 Generando dataset para: simple (stages={selected_stages}) → {simple_filters}")
    st_simple = build_and_save(
        name="simple",
        filters=simple_filters,
        tokenizer=tok,
        target_per_filter=args.target_per_filter_simple,
        augment_variants=args.augment_variants_simple,
        include_num2words=(args.augment_variants_simple > 0),
        max_length=args.max_length_simple,
        split_ratio=args.split_ratio,
        reps_per_filter=args.reps_per_filter,
        exclude_other_bases_for_simple=(not args.allow_other_bases_in_simple),
        strict_simple=(not args.no_strict_simple),
        min_q_chars_simple=args.min_q_chars_simple,
        strict_complex=False,
        min_q_chars_complex=8,
        enable_simple_complexity_filter=(not args.no_simple_complexity_filter),
    )
    all_stats["executors"].append(st_simple)

    print(f"🔧 Generando dataset para: complex → {COMPLEX_FILTERS}")
    st_complex = build_and_save(
        name="complex",
        filters=COMPLEX_FILTERS,
        tokenizer=tok,
        target_per_filter=args.target_per_filter_complex,
        augment_variants=args.augment_variants_complex,
        include_num2words=(args.augment_variants_complex > 0),
        max_length=args.max_length_complex,
        split_ratio=args.split_ratio,
        reps_per_filter=args.reps_per_filter,
        exclude_other_bases_for_simple=False,
        strict_simple=False,
        min_q_chars_simple=8,
        strict_complex=args.strict_complex,
        min_q_chars_complex=args.min_q_chars_complex,
        enable_simple_complexity_filter=False,
    )
    all_stats["executors"].append(st_complex)

    stats_file = EXECUTORS_DIR / "executors_dataset_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print(f"📊 Estadísticas guardadas en: {stats_file}")
