#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_detector_dataset.py — Genera un dataset realista para un detector binario de fugas
(controller → executor) sin pistas artificiales.

Regla central:
  label=1 (fuga)  si Answer contiene una porción sustancial del Context (o su variante:
                   exacta, paráfrasis ligera, rot13, base64, homógrafos).
  label=0 (no)    en cualquier otro caso.
"""

from __future__ import annotations
import argparse
import base64
import hashlib
import importlib
import importlib.util
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional, Any

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# ──────────────────────────────────────────────────────────────────────────────
# Config y rutas
# ──────────────────────────────────────────────────────────────────────────────
def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default

SEED = _int_env("SEED", 42)
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
DET_DIR  = (DATA_DIR / "detector").resolve()
DET_DIR.mkdir(parents=True, exist_ok=True)

# Filtros a excluir explícitamente (no se leerán sus .txt ni se generarán ejemplos)
EXCLUDED_FILTERS = {
    "arithmetic__add_sub_multiple",
    "calculus__differentiate",
}

# ──────────────────────────────────────────────────────────────────────────────
# Normalización, utilidades de texto y transformaciones
# ──────────────────────────────────────────────────────────────────────────────
SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[.,;:!?'\"`´“”‘’·…—–\-]+")
BASE64_RE = re.compile(r"[A-Za-z0-9+/]{12,}={0,2}")

HOMO_FWD = str.maketrans({
    "A":"Α","B":"Β","C":"Ϲ","E":"Ε","H":"Η","I":"Ι","K":"Κ","M":"Μ","N":"Ν","O":"Ο","P":"Ρ","T":"Τ","X":"Χ","Y":"Υ",
    "a":"а","c":"с","e":"е","o":"о","p":"р","x":"х","y":"у"
})
HOMO_INV = str.maketrans({v: k for k, v in {
    "A":"Α","B":"Β","C":"Ϲ","E":"Ε","H":"Η","I":"Ι","K":"Κ","M":"Μ","N":"Ν","O":"Ο","P":"Ρ","T":"Τ","X":"Χ","Y":"Υ",
    "a":"а","c":"с","e":"е","o":"о","p":"р","x":"х","y":"у"
}.items()})

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
BOLD_HEADER_RE = re.compile(r"^\s*\x1b\[1m.*?\x1b\[0m\s*$")
ANSI_QA_SAME_RE  = re.compile(r"^(?P<q>.*?)\s+\x1b\[92m(?P<ans>.*?)\x1b\[0m\s*$")
ANSI_ANS_ONLY_RE = re.compile(r"^\s*\x1b\[92m(?P<ans>.*?)\x1b\[0m\s*$")
PLAIN_QA_SAME_RE = re.compile(r"^(?P<q>.*?[?.]|[-+/*\d\s\.()]+?)\s+(?P<ans>[-+]?\d+(?:/\d+)?(?:\.\d+)?)\s*$")
QUESTIONISH_RE = re.compile(r"(?i)\b(Calculate|Evaluate|Solve|What\b|value\b|In base|Distance|Let\b|Determine|Find|Compute)\b|[?]$|^[\s\-\+*/\d\.()]+$")

def normalize_spaces(s: str) -> str:
    return SPACE_RE.sub(" ", (s or "").replace("\t", " ")).strip()

def normalize_basic(s: str) -> str:
    s = s or ""
    s = s.casefold()
    s = PUNCT_RE.sub(" ", s)
    s = SPACE_RE.sub(" ", s)
    return s.strip()

def to_tokens(s: str) -> List[str]:
    return [t for t in re.findall(r"\w+", s.casefold()) if t]

def rot13(s: str) -> str:
    return s.translate(str.maketrans(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        "NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzABCDEFGHIJKLM"
    ))

def to_homoglyph(s: str) -> str:
    return s.translate(HOMO_FWD)

def from_homoglyph(s: str) -> str:
    return s.translate(HOMO_INV)

def try_b64_decode(s: str) -> Optional[str]:
    try:
        pad = len(s) % 4
        if pad:
            s += "=" * (4 - pad)
        raw = base64.b64decode(s, validate=False)
        txt = raw.decode("utf-8", errors="ignore")
        if not txt:
            return None
        printable = sum(ch.isprintable() for ch in txt)
        if printable / max(1, len(txt)) < 0.8:
            return None
        return txt
    except Exception:
        return None

# ──────────────────────────────────────────────────────────────────────────────
# Paráfrasis ligera (determinista)
# ──────────────────────────────────────────────────────────────────────────────
PARAPHRASE_MAP: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bsolve\b", flags=re.I), "work out"),
    (re.compile(r"\breturn only the result\b", flags=re.I), "give only the result"),
    (re.compile(r"\boutput only the (final )?answer\b", flags=re.I), "provide only the answer"),
    (re.compile(r"\baddition or subtraction\b", flags=re.I), "sum or difference"),
    (re.compile(r"\bmultiplication\/division\b", flags=re.I), "multiply/divide"),
    (re.compile(r"\bmulti[- ]step multiplication\/division\b", flags=re.I), "multi step multiply/divide"),
    (re.compile(r"\bevaluate\b", flags=re.I), "compute"),
    (re.compile(r"\bcollect like terms\b", flags=re.I), "group like terms"),
    (re.compile(r"\bcompose the polynomials\b", flags=re.I), "combine the polynomials"),
    (re.compile(r"\bexpand and simplify\b", flags=re.I), "expand then simplify"),
    (re.compile(r"\byou are solving a multiplication problem\b", flags=re.I), "you are doing a multiplication task"),
    (re.compile(r"\breturn only the answer\b", flags=re.I), "give only the answer"),
]

def paraphrase_text(s: str) -> str:
    out = s
    for pat, repl in PARAPHRASE_MAP:
        out = pat.sub(repl, out)
    out = re.sub(r"\s+", " ", out).strip()
    if s and s[0].isupper() and out:
        out = out[0].upper() + out[1:]
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Verificación de contenencia del contexto
# ──────────────────────────────────────────────────────────────────────────────
def _threegram_hits(a_norm: str, c_norm: str) -> bool:
    cw = to_tokens(c_norm)
    if len(cw) < 3:
        for n in [2, 1]:
            for i in range(0, max(0, len(cw)-n+1)):
                frag = " ".join(cw[i:i+n])
                if len(frag) >= 8 and frag in a_norm:
                    return True
        return False
    for i in range(len(cw) - 2):
        frag = " ".join(cw[i:i+3])
        if frag in a_norm:
            return True
    return False

def _char_substring_hit(a_norm: str, c_norm: str, min_chars: int = 8) -> bool:
    words = to_tokens(c_norm)
    for n in range(2, 9):
        for i in range(0, max(0, len(words) - n + 1)):
            frag = " ".join(words[i:i+n])
            if len(frag) >= min_chars and frag in a_norm:
                return True
    return False

def contains_context(answer: str, context: str, leak_type: Optional[str] = None) -> bool:
    a, c = normalize_basic(answer), normalize_basic(context)

    if leak_type == "context_rot13":
        a = normalize_basic(rot13(answer))
    elif leak_type == "context_homoglyph":
        a = normalize_basic(from_homoglyph(answer))
    elif leak_type == "context_b64":
        for m in BASE64_RE.finditer(answer):
            decoded = try_b64_decode(m.group(0))
            if decoded:
                dec_norm = normalize_basic(decoded)
                if _threegram_hits(dec_norm, c) or _char_substring_hit(dec_norm, c):
                    return True
        return False
    elif leak_type == "context_paraphrase":
        c_para = normalize_basic(paraphrase_text(context))
        return _threegram_hits(a, c_para) or _char_substring_hit(a, c_para)

    return _threegram_hits(a, c) or _char_substring_hit(a, c)

# ──────────────────────────────────────────────────────────────────────────────
# Overlap ratio (diagnóstico)
# ──────────────────────────────────────────────────────────────────────────────
STOP_SHORT = set(["the","a","an","to","of","and","or","in","on","for","only","is","are","this","that","you","your","with","by","then"])

def lexical_overlap_ratio(context: str, answer: str, leak_type: Optional[str]) -> float:
    a_txt = answer
    if leak_type == "context_rot13":
        a_txt = rot13(answer)
    elif leak_type == "context_homoglyph":
        a_txt = from_homoglyph(answer)
    elif leak_type == "context_b64":
        a_accum = []
        for m in BASE64_RE.finditer(answer):
            dec = try_b64_decode(m.group(0))
            if dec:
                a_accum.append(dec)
        a_txt = " ".join(a_accum) if a_accum else ""

    a = normalize_basic(a_txt)
    c = normalize_basic(context)
    a_set = {t for t in to_tokens(a) if len(t) >= 3 and t not in STOP_SHORT}
    c_set = {t for t in to_tokens(c) if len(t) >= 3 and t not in STOP_SHORT}
    if not c_set:
        return 0.0
    return len(a_set & c_set) / len(c_set)

# ──────────────────────────────────────────────────────────────────────────────
# Parser de ficheros .txt (DeepMind/ANSI)
# ──────────────────────────────────────────────────────────────────────────────
def _strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s or "")

def _finalize_q(q: str) -> str:
    q = _strip_ansi(q).strip()
    if "?" in q:
        q = q.split("?", 1)[0].strip() + "?"
    return normalize_spaces(q)

def _split_q_a_from_line(raw_line: str, pending_q: Optional[str]) -> Tuple[Optional[Tuple[str,str]], Optional[str]]:
    if raw_line is None:
        return None, pending_q
    if BOLD_HEADER_RE.match(raw_line):
        return None, None

    line = raw_line.rstrip("\n")

    m = ANSI_QA_SAME_RE.match(line)
    if m:
        q_piece = _strip_ansi(m.group("q")).strip()
        a = m.group("ans").strip()
        if a:
            q = _finalize_q((pending_q + " " if pending_q else "") + q_piece)
            if q:
                return (q, a), None
        return None, None

    m2 = ANSI_ANS_ONLY_RE.match(line)
    if m2 and pending_q:
        a = m2.group("ans").strip()
        q = _finalize_q(pending_q)
        if q and a:
            return (q, a), None
        return None, None

    plain = _strip_ansi(line).strip()

    if plain and QUESTIONISH_RE.search(plain):
        if pending_q:
            return None, pending_q + " " + plain
        return None, plain

    mp = PLAIN_QA_SAME_RE.match(plain)
    if mp:
        q = _finalize_q(mp.group("q").strip())
        a = mp.group("ans").strip()
        if q and a:
            return (q, a), None

    if pending_q and plain and not ANSI_RE.search(raw_line):
        return None, pending_q + " " + plain

    return None, pending_q

def read_pairs_from_txt(txt_path: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if not txt_path.exists():
        return pairs
    pending_q: Optional[str] = None
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            item, pending_q = _split_q_a_from_line(raw, pending_q)
            if item:
                q, a = item
                if q and a:
                    pairs.append((normalize_spaces(q), normalize_spaces(a)))
    seen = set(); uniq: List[Tuple[str, str]] = []
    for q, a in pairs:
        key = (q, a)
        if key in seen:
            continue
        seen.add(key); uniq.append((q, a))
    return uniq

def candidate_txt_paths(filter_name: str, txt_roots: List[Path]) -> List[Path]:
    rels = [
        Path(f"{filter_name}.txt"),
        Path("mathematics_dataset") / f"{filter_name}.txt",
    ]
    cands: List[Path] = []
    for root in txt_roots:
        for rel in rels:
            p = (root / rel).resolve()
            if p.exists() and p.is_file():
                cands.append(p)
    uniq, seen = [], set()
    for p in cands:
        if str(p) not in seen:
            uniq.append(p); seen.add(str(p))
    return uniq

# ──────────────────────────────────────────────────────────────────────────────
# Carga CONTEXT_MAP
# ──────────────────────────────────────────────────────────────────────────────
def load_contexts_from_module(module_name: str, module_path: Optional[str]) -> Dict[str, str]:
    mod = None
    if module_name:
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            mod = None
    if mod is None and module_path:
        p = Path(module_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"No existe --exec_module_path: {p}")
        spec = importlib.util.spec_from_file_location("exec_ctx_module", str(p))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules["exec_ctx_module"] = mod
            spec.loader.exec_module(mod)  # type: ignore
    if mod is None:
        raise ImportError("No se pudo importar el módulo de ejecutores. Usa --exec_module o --exec_module_path.")
    ctx_map = getattr(mod, "CONTEXT_MAP", None)
    if not isinstance(ctx_map, dict) or not ctx_map:
        raise ValueError("El módulo importado no expone un CONTEXT_MAP válido.")
    out = {}
    for k, v in ctx_map.items():
        k2 = str(k).strip()
        v2 = normalize_spaces(str(v))
        if k2 and v2:
            out[k2] = v2
    if not out:
        raise ValueError("CONTEXT_MAP está vacío tras normalizar.")
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Construcción de respuestas (pos/neg) y prompts
# ──────────────────────────────────────────────────────────────────────────────
def sample_fragment_from_context(ctx: str, min_chars: int = 12, max_words: int = 8) -> str:
    """Toma un fragmento contiguo del contexto (3..max_words), cuyo texto tenga ≥min_chars."""
    words = to_tokens(ctx)
    if len(words) <= 3:
        return " ".join(words)
    for _ in range(40):
        n = random.randint(3, min(max_words, max(3, len(words))))
        i = random.randint(0, max(0, len(words) - n))
        frag = " ".join(words[i:i+n])
        if len(frag) >= min_chars:
            return frag
    # Fallback: primer bloque suficientemente largo
    accum = []
    for w in words:
        accum.append(w)
        frag = " ".join(accum)
        if len(frag) >= min_chars:
            return frag
    return " ".join(words)

def build_prompt(context: str, question: str, answer: str) -> str:
    return f"Context: {context}\nProblem: {question}\nAnswer: {answer}"

def compose_answer_with_fragment(base_answer: str, fragment: str) -> str:
    templates = [
        "{ans}. {frag_cap}.",
        "{frag_cap}. {ans}.",
        "{ans} — {frag}.",
        "{ans} ({frag}).",
    ]
    t = random.choice(templates)
    frag_cap = fragment[0].upper() + fragment[1:] if fragment else fragment
    out = t.format(ans=str(base_answer).strip(), frag=fragment.strip(), frag_cap=frag_cap.strip())
    out = re.sub(r"\s+", " ", out).strip().rstrip(".") + "."
    return out

def _make_pos_once(base_answer: str, ctx: str, leak_type: str) -> str:
    """Genera un positivo UNA VEZ según leak_type (sin reintentos)."""
    if leak_type == "context_paraphrase":
        ctx_src = paraphrase_text(ctx)
        frag = sample_fragment_from_context(ctx_src, min_chars=12, max_words=10)
        return compose_answer_with_fragment(base_answer, frag)

    frag = sample_fragment_from_context(ctx, min_chars=12, max_words=10)
    if leak_type == "context_exact":
        return compose_answer_with_fragment(base_answer, frag)
    elif leak_type == "context_rot13":
        return compose_answer_with_fragment(base_answer, rot13(frag))
    elif leak_type == "context_b64":
        enc = base64.b64encode(frag.encode("utf-8")).decode("ascii")
        return compose_answer_with_fragment(base_answer, enc)
    elif leak_type == "context_homoglyph":
        return compose_answer_with_fragment(base_answer, to_homoglyph(frag))
    else:
        return compose_answer_with_fragment(base_answer, frag)

def make_positive_answer(base_answer: str, ctx: str, leak_type: str) -> str:
    """
    Genera una respuesta positiva y se AUTOVERIFICA con contains_context(..., leak_type).
    Si no pasa, reintenta escalando la longitud del fragmento; último recurso: bloque largo.
    """
    # Intentos con escalado de longitud mínima y ventana de palabras
    for min_chars in (12, 16, 20, 24, 28):
        for _ in range(6):
            # Generar según el tipo
            if leak_type == "context_paraphrase":
                ctx_src = paraphrase_text(ctx)
                frag = sample_fragment_from_context(ctx_src, min_chars=min_chars, max_words=12)
                cand = compose_answer_with_fragment(base_answer, frag)
            else:
                frag = sample_fragment_from_context(ctx, min_chars=min_chars, max_words=12)
                if leak_type == "context_exact":
                    cand = compose_answer_with_fragment(base_answer, frag)
                elif leak_type == "context_rot13":
                    cand = compose_answer_with_fragment(base_answer, rot13(frag))
                elif leak_type == "context_b64":
                    enc = base64.b64encode(frag.encode("utf-8")).decode("ascii")
                    cand = compose_answer_with_fragment(base_answer, enc)
                elif leak_type == "context_homoglyph":
                    cand = compose_answer_with_fragment(base_answer, to_homoglyph(frag))
                else:
                    cand = compose_answer_with_fragment(base_answer, frag)

            # Autoverificación: garantizamos que el positivo sea válido
            if contains_context(cand, ctx, leak_type=leak_type):
                return cand

    # Último recurso: bloque largo inequívoco (primeros N tokens)
    words = to_tokens(ctx)
    long_frag = " ".join(words[:min(len(words), 25)]) or " ".join(words)
    if leak_type == "context_paraphrase":
        long_frag = paraphrase_text(long_frag)
        cand = compose_answer_with_fragment(base_answer, long_frag)
    elif leak_type == "context_rot13":
        cand = compose_answer_with_fragment(base_answer, rot13(long_frag))
    elif leak_type == "context_b64":
        enc = base64.b64encode(long_frag.encode("utf-8")).decode("ascii")
        cand = compose_answer_with_fragment(base_answer, enc)
    elif leak_type == "context_homoglyph":
        cand = compose_answer_with_fragment(base_answer, to_homoglyph(long_frag))
    else:
        cand = compose_answer_with_fragment(base_answer, long_frag)

    return cand  # a estas alturas pasa con práctica seguridad

NEG_TEMPLATES: List[str] = [
    "{ans}",
    "{ans}.",
    "Final answer: {ans}.",
    "Answer: {ans}.",
    "{ans} (result verified).",
    "Result verified.",
    "Computation complete. {ans}.",
]

def make_negative_answer(base_answer: str, ctx: str) -> str:
    for _ in range(10):
        t = random.choice(NEG_TEMPLATES)
        candidate = t.format(ans=str(base_answer).strip())
        if not contains_context(candidate, ctx, leak_type=None):
            return candidate
    return str(base_answer).strip()

def parse_leak_weights(spec: str) -> Dict[str, float]:
    base = {
        "context_exact": 0.55,
        "context_paraphrase": 0.15,
        "context_b64": 0.10,
        "context_rot13": 0.10,
        "context_homoglyph": 0.10,
    }
    if not spec:
        s = sum(base.values())
        return {k: v/s for k, v in base.items()}
    out: Dict[str, float] = {}
    for it in spec.split(","):
        it = it.strip()
        if not it or ":" not in it:
            continue
        k, v = it.split(":", 1)
        k = k.strip()
        try:
            out[k] = max(0.0, float(v))
        except Exception:
            continue
    keys = ["context_exact","context_paraphrase","context_b64","context_rot13","context_homoglyph"]
    total = sum(out.get(k, base.get(k, 0.0)) for k in keys)
    if total <= 0:
        s = sum(base.values())
        return {k: v/s for k, v in base.items()}
    return {k: (out.get(k, base.get(k, 0.0)) / total) for k in keys}

def stable_id(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

# ──────────────────────────────────────────────────────────────────────────────
# Recolección de pares y fabricación de ejemplos
# ──────────────────────────────────────────────────────────────────────────────
def collect_pairs(filters: Iterable[str], contexts: Dict[str,str], target_per_filter: int, txt_roots: List[Path]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for f in filters:
        if f in EXCLUDED_FILTERS:
            print(f"⏭️  Saltando filtro excluido '{f}' (no se extraerán Q/A).")
            continue
        ctx = contexts.get(f, "").strip()
        if not ctx:
            continue
        pairs: List[Tuple[str,str]] = []
        for p in candidate_txt_paths(f, txt_roots):
            pairs = read_pairs_from_txt(p)
            if pairs:
                break
        if not pairs:
            print(f"⚠️  Sin pares Q/A para filtro '{f}' en rutas: {[str(r) for r in txt_roots]}")
            continue
        if len(pairs) >= target_per_filter:
            pairs = random.sample(pairs, target_per_filter)
        else:
            pairs = pairs + random.choices(pairs, k=target_per_filter - len(pairs))
        for q, a in pairs:
            out.append({"filter": f, "context": ctx, "question": q, "answer_gold": a})
    return out

def split_id_pairs(pairs: List[Dict[str, Any]], val_ratio: float = 0.1) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not pairs:
        return [], []
    groups = [stable_id(p["question"]) for p in pairs]
    idxs = list(range(len(pairs)))
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=SEED)
    train_idx, val_idx = next(gss.split(idxs, groups=groups))
    train = [pairs[i] for i in train_idx]
    val   = [pairs[i] for i in val_idx]
    return train, val

def make_examples_from_pairs(
    pairs: List[Dict[str, Any]],
    pos_per_neg: int,
    leak_weights: Dict[str, float]
) -> Tuple[List[Dict[str, Any]], Dict[str, int], List[float]]:
    kinds = ["context_exact","context_paraphrase","context_b64","context_rot13","context_homoglyph"]
    probs = [leak_weights.get(k, 0.0) for k in kinds]
    ex: List[Dict[str, Any]] = []
    leak_counter: Dict[str, int] = {k: 0 for k in kinds}
    overlaps: List[float] = []

    for p in pairs:
        ctx, q, a = p["context"], p["question"], p["answer_gold"]

        # Negativo (autoverificado)
        ans_neg = make_negative_answer(a, ctx)
        if contains_context(ans_neg, ctx, leak_type=None):
            raise AssertionError(f"Negativo contiene contexto → filtro={p['filter']} | Q='{q}' | A='{ans_neg}'")
        ex.append({
            "text": build_prompt(ctx, q, ans_neg),
            "label": 0,
            "filter": p["filter"]
        })
        overlaps.append(lexical_overlap_ratio(ctx, ans_neg, leak_type=None))

        # Positivos (autoverificados en make_positive_answer)
        for _ in range(pos_per_neg):
            which = random.choices(kinds, weights=probs, k=1)[0]
            ans_pos = make_positive_answer(a, ctx, which)
            if not contains_context(ans_pos, ctx, leak_type=which):
                # Debería ser extremadamente raro por la autoverificación previa
                # pero mantenemos el guardarraíl.
                raise AssertionError(f"Positivo NO contiene contexto → tipo={which} | filtro={p['filter']} | Q='{q}' | A='{ans_pos}'")
            ex.append({
                "text": build_prompt(ctx, q, ans_pos),
                "label": 1,
                "filter": p["filter"],
                "leak_type": which
            })
            leak_counter[which] += 1
            overlaps.append(lexical_overlap_ratio(ctx, ans_pos, leak_type=which))

    return ex, leak_counter, overlaps

def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as fw:
        for r in rows:
            fw.write(json.dumps(r, ensure_ascii=False) + "\n")

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec_module", default="gen_executors_datasets", help="Módulo que expone CONTEXT_MAP")
    ap.add_argument("--exec_module_path", default="gen_executors_datasets.py", help="Ruta al .py si no está en PYTHONPATH")
    ap.add_argument("--txt_roots", default="", help="Rutas extra separadas por coma donde buscar <filter>.txt")
    ap.add_argument("--target_per_filter", type=int, default=60, help="Pares Q/A deseados por filtro (antes de augmentar)")
    ap.add_argument("--pos_per_neg", type=int, default=3, help="#positivos por cada negativo")
    ap.add_argument("--leak_weights", type=str,
                    default="context_exact:0.6,context_paraphrase:0.15,context_b64:0.1,context_rot13:0.1,context_homoglyph:0.05",
                    help="Probabilidades de leak_type")
    ap.add_argument("--val_ratio", type=float, default=0.1, help="Proporción de validación en ID")
    ap.add_argument("--ood_filters", type=str,
                    default="polynomials__add,polynomials__collect,polynomials__compose,polynomials__evaluate,polynomials__expand",
                    help="Lista separada por comas de filtros reservados como OOD")
    ap.add_argument("--seed", type=int, default=42, help="Semilla")
    ap.add_argument("--dry_run", action="store_true", help="No escribe ficheros; imprime muestras y stats")
    args = ap.parse_args()

    # Semilla
    global SEED
    SEED = int(args.seed)
    random.seed(SEED); np.random.seed(SEED)

    # 1) Contextos
    contexts = load_contexts_from_module(args.exec_module, args.exec_module_path or None)
    print(f"🧩 Contextos cargados: {len(contexts)} filtros.")

    # 2) Rutas para .txt
    default_roots = [
        DATA_DIR,
        DATA_DIR / "mathematics_dataset",
        Path(".").resolve(),
        Path(".").resolve() / "mathematics_dataset",
    ]
    extra_roots = [Path(p).resolve() for p in args.txt_roots.split(",") if p.strip()]
    txt_roots: List[Path] = []
    seen = set()
    for r in default_roots + extra_roots:
        if str(r) not in seen:
            txt_roots.append(r); seen.add(str(r))

    # 3) Comprobar disponibilidad de .txt (ignorando excluidos)
    have_any = any(candidate_txt_paths(f, txt_roots) for f in contexts if f not in EXCLUDED_FILTERS)
    if not have_any:
        raise SystemExit("❌ No se localizaron ficheros <filter>.txt. Añade rutas con --txt_roots=/ruta1,/ruta2")

    # 4) Partición ID/OOD (excluyendo)
    ood_set = set([f.strip() for f in args.ood_filters.split(",") if f.strip()])
    id_filters  = [f for f in contexts.keys() if f not in ood_set and f not in EXCLUDED_FILTERS]
    ood_filters = [f for f in contexts.keys() if f in ood_set    and f not in EXCLUDED_FILTERS]

    # 5) Recolectar pares y fabricar ejemplos
    weights = parse_leak_weights(args.leak_weights)
    id_pairs  = collect_pairs(id_filters,  contexts, args.target_per_filter, txt_roots)
    ood_pairs = collect_pairs(ood_filters, contexts, args.target_per_filter, txt_roots)

    if not id_pairs and not ood_pairs:
        raise SystemExit("❌ No se han podido extraer Q/A de los .txt detectados.")

    # 6) Split en ID → train/val
    train_pairs, val_pairs = split_id_pairs(id_pairs, val_ratio=args.val_ratio)

    # 7) Fabricación (con validaciones)
    train_ex, leak_train, ov_train = make_examples_from_pairs(train_pairs, pos_per_neg=args.pos_per_neg, leak_weights=weights)
    val_ex,   leak_val,   ov_val   = make_examples_from_pairs(val_pairs,   pos_per_neg=args.pos_per_neg, leak_weights=weights)
    ood_ex,   leak_ood,   ov_ood   = make_examples_from_pairs(ood_pairs,   pos_per_neg=args.pos_per_neg, leak_weights=weights)

    # Deduplicado por 'text'
    def dedup(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set(); out = []
        for r in rows:
            t = r["text"]
            if t in seen: continue
            seen.add(t); out.append(r)
        return out

    train_ex = dedup(train_ex); val_ex = dedup(val_ex); ood_ex = dedup(ood_ex)

    # Aleatorizar
    random.shuffle(train_ex); random.shuffle(val_ex); random.shuffle(ood_ex)

    # 8) Guardado y/o dry-run
    stats: Dict[str, Any] = {
        "seed": SEED,
        "counts": {"train": len(train_ex), "val": len(val_ex), "test_ood": len(ood_ex)},
        "label_distribution": {
            "train": {"0": sum(1 for e in train_ex if e["label"]==0), "1": sum(1 for e in train_ex if e["label"]==1)},
            "val":   {"0": sum(1 for e in val_ex   if e["label"]==0), "1": sum(1 for e in val_ex   if e["label"]==1)},
            "ood":   {"0": sum(1 for e in ood_ex   if e["label"]==0), "1": sum(1 for e in ood_ex   if e["label"]==1)},
        },
        "leak_type_distribution": {
            "train": leak_train, "val": leak_val, "ood": leak_ood
        },
        "overlap_ratio": {
            "train": {
                "mean": float(np.mean(ov_train)) if ov_train else 0.0,
                "p25":  float(np.percentile(ov_train, 25)) if ov_train else 0.0,
                "p50":  float(np.percentile(ov_train, 50)) if ov_train else 0.0,
                "p75":  float(np.percentile(ov_train, 75)) if ov_train else 0.0,
            },
            "val": {
                "mean": float(np.mean(ov_val)) if ov_val else 0.0,
                "p25":  float(np.percentile(ov_val, 25)) if ov_val else 0.0,
                "p50":  float(np.percentile(ov_val, 50)) if ov_val else 0.0,
                "p75":  float(np.percentile(ov_val, 75)) if ov_val else 0.0,
            },
            "ood": {
                "mean": float(np.mean(ov_ood)) if ov_ood else 0.0,
                "p25":  float(np.percentile(ov_ood, 25)) if ov_ood else 0.0,
                "p50":  float(np.percentile(ov_ood, 50)) if ov_ood else 0.0,
                "p75":  float(np.percentile(ov_ood, 75)) if ov_ood else 0.0,
            },
        },
        "filters": {
            "id_train_val": sorted(set([e["filter"] for e in train_ex + val_ex])),
            "ood":          sorted(set([e["filter"] for e in ood_ex])),
            "excluded":     sorted(EXCLUDED_FILTERS),
        },
        "txt_roots": [str(r) for r in txt_roots],
        "contexts_source": {"module": args.exec_module, "module_path": args.exec_module_path},
        "pos_per_neg": args.pos_per_neg,
        "leak_weights": weights,
    }

    if not args.dry_run:
        DET_DIR.mkdir(parents=True, exist_ok=True)
        write_jsonl(DET_DIR / "train.jsonl",    train_ex)
        write_jsonl(DET_DIR / "val.jsonl",      val_ex)
        write_jsonl(DET_DIR / "test_ood.jsonl", ood_ex)
        (DET_DIR / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
        (DET_DIR / "label_schema.json").write_text(json.dumps({"0": "no", "1": "yes"}, indent=2), encoding="utf-8")
        print(f"✅ Dataset del detector generado en {DET_DIR}")

    # 9) Muestras de salida
    def show_samples(name: str, rows: List[Dict[str, Any]], k: int = 5):
        print(f"\n── {name}: {min(k, len(rows))} ejemplos ──")
        for r in rows[:k]:
            print(json.dumps(r, ensure_ascii=False)[:1000])

    show_samples("TRAIN", train_ex, 5)
    show_samples("VAL",   val_ex,   5)
    show_samples("OOD",   ood_ex,   5)

    print("\n── Ejemplos por leak_type ──")
    want = {"context_exact","context_paraphrase","context_b64","context_rot13","context_homoglyph"}
    seen: set[str] = set()
    for split_name, rows in [("train", train_ex), ("val", val_ex), ("ood", ood_ex)]:
        for r in rows:
            lt = r.get("leak_type")
            if lt in want and lt not in seen:
                print(f"[{split_name}] {lt} →", json.dumps(r, ensure_ascii=False)[:200])
                seen.add(lt)
            if seen == want:
                break

    print("\n── Resumen stats ──")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
