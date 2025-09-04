#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_executors.py — QLoRA 4-bit con EM real (parches attention_mask + decode seguro):
- EM del trainer = EM de inferencia (sin leakage de labels)
- Predicción evaluada generando desde el prefijo hasta 'Answer' (misma lógica que manual_eval_em)
- AllowedTokensLogitsProcessor en 'simple'
- ✅ Se pasa attention_mask explícito en TODAS las generate(...)
- ✅ compute_metrics decodifica predicciones convirtiéndolas a list[int] (evita OverflowError)
"""

from __future__ import annotations
import argparse, gc, json, math, os, random, re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    Seq2SeqTrainingArguments, TrainerCallback, EvalPrediction,
    EarlyStoppingCallback, default_data_collator, set_seed
)
# compat logits processors
try:
    from transformers import LogitsProcessorList
    from transformers.generation.logits_process import LogitsProcessor
except Exception:
    from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from transformers import Seq2SeqTrainer
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig

from fractions import Fraction
from decimal import Decimal, InvalidOperation

# ───────── Config ─────────
SEED = int(os.getenv("SEED", 42))
set_seed(SEED)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DATA_DIR      = Path(os.getenv("DATA_DIR", "./data")).resolve()
EXECUTORS_DIR = DATA_DIR / "executors"; EXECUTORS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR   = Path("./models/executors/results"); RESULTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

# ───────── Carga 4-bit ─────────
def load_llama_safe(base: str, token: str):
    tok = AutoTokenizer.from_pretrained(base, use_fast=True, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Decoder-only: el padding a la IZQUIERDA ayuda si alguna vez paddeas prefijos
    tok.padding_side = "left"

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base, token=token, quantization_config=bnb_cfg,
        torch_dtype=compute_dtype, device_map="auto", low_cpu_mem_usage=True,
    )
    model.generation_config.pad_token_id = tok.eos_token_id
    model.generation_config.eos_token_id = tok.eos_token_id
    return model, tok

# ───────── Canon numérica ─────────
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
SPACE_RE = re.compile(r"\s+")
FRACTION_RE = re.compile(r"^\s*([-+]?\d+)\s*/\s*([-+]?\d+)\s*$")
SAFE_EXPR = re.compile(r"^[\s\d\./\+\-]+$")
_ANSWER_ANCHOR_RE = re.compile(r"(?i)(?:^|\n)answer\s*:?", re.IGNORECASE)

def _normalize_fraction(ans: str) -> str:
    m = FRACTION_RE.match(ans or "")
    if not m: return (ans or "").strip()
    num, den = int(m.group(1)), int(m.group(2))
    if den == 0: return (ans or "").strip()
    from math import gcd
    g = gcd(num, den); num//=g; den//=g
    if den < 0: den=-den; num=-num
    return f"{num}/{den}" if den != 1 else str(num)

def _to_canonical_number(s: str) -> Optional[str]:
    s = (s or "").strip()
    m = FRACTION_RE.match(s)
    if m:
        try:
            f = Fraction(int(m.group(1)), int(m.group(2)))
            return f"{f.numerator}/{f.denominator}" if f.denominator != 1 else str(f.numerator)
        except Exception:
            return None
    if SAFE_EXPR.match(s) and any(op in s for sgn, op in enumerate(['+','-','/',' ']) if s.count(op) > 0):
        try:
            total = Fraction(0, 1)
            s_norm = re.sub(r"\s+", " ", s).replace("+-", "-").replace("--", "+")
            for term in re.findall(r"[+\-]?\s*\d+(?:/\d+)?(?:\.\d+)?", s_norm):
                term = term.replace(" ", "")
                if '/' in term and '.' not in term:
                    total += Fraction(term)
                else:
                    total += Fraction(Decimal(term))
            return f"{total.numerator}/{total.denominator}" if total.denominator != 1 else str(total.numerator)
        except (InvalidOperation, ZeroDivisionError, ValueError):
            return None
    try:
        d = Decimal(s)
        out = format(d.normalize(), 'f')
        if '.' in out: out = out.rstrip('0').rstrip('.')
        return out if out else "0"
    except InvalidOperation:
        return None

def _first_numeric_chunk(text: str) -> str:
    text = (text or "").strip().split("\n", 1)[0].strip()
    m0 = re.match(r"^[\s\d\./\+\-]+", text)
    if m0: return m0.group(0).strip()
    m1 = re.search(r"[+\-]?\d+(?:/\d+)?(?:\.\d+)?", text)
    return m1.group(0) if m1 else text

def canonify_text_answer(text: str) -> str:
    s = ANSI_RE.sub("", (text or "").strip())
    s = re.sub(SPACE_RE, " ", s).strip()
    s = _normalize_fraction(s)
    s = _first_numeric_chunk(s)
    out = _to_canonical_number(s)
    return (out or "").strip()

def _extract_after_answer_marker(text: str) -> str:
    last = None
    for m in _ANSWER_ANCHOR_RE.finditer(text or ""):
        last = m
    return (text[last.end():] if last else (text or "")).strip()

# ───────── Utils plots ─────────
def detect_precision() -> Tuple[bool, bool]:
    if not torch.cuda.is_available():
        print("⚠️  No GPU — entrenamiento en CPU"); return False, False
    return (not torch.cuda.is_bf16_supported()), torch.cuda.is_bf16_supported()

def ema_smooth(values, weight=0.6):
    if not values: return values
    out = [values[0]]
    for v in values[1:]:
        out.append(weight * out[-1] + (1 - weight) * v)
    return out

class TrainPerplexityCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs and "eval_loss" not in logs:
            logs["train_perplexity"] = math.exp(min(float(logs["loss"]), 20))

def _annotate_final(ax, x_vals, y_vals, label_fmt: str):
    if x_vals and y_vals:
        ax.annotate(label_fmt.format(y_vals[-1]),
                    xy=(x_vals[-1], y_vals[-1]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=9, fontweight="bold")

def plot_logs(logs: List[Dict[str, Any]], name: str, smooth_w: float = 0.6):
    tr_s, tr_l, tr_p = [], [], []
    ev_s, ev_l, ev_p = [], [], []
    for entry in logs:
        if "step" not in entry: continue
        step = entry["step"]
        if "loss" in entry and "eval_loss" not in entry:
            tr_s.append(step); tr_l.append(entry["loss"])
            tr_p.append(math.exp(min(float(entry["loss"]), 20)))
        if "eval_loss" in entry:
            ev_s.append(step); ev_l.append(entry["eval_loss"])
            ppl = entry.get("eval_perplexity")
            if ppl is None and entry.get("eval_loss") is not None:
                ppl = math.exp(min(float(entry["eval_loss"]), 20))
            ev_p.append(ppl)

    tr_l_s = ema_smooth(tr_l, smooth_w); tr_p_s = ema_smooth(tr_p, smooth_w)
    ev_l_s = ema_smooth(ev_l, smooth_w); ev_p_s = ema_smooth(ev_p, smooth_w)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f"Training Metrics – Executor «{name}»")
    if tr_s: ax1.plot(tr_s, tr_l_s, label="Train Loss")
    if ev_s: ax1.plot(ev_s, ev_l_s, label="Eval Loss")
    ax1.set_xlabel("Step"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)
    if ev_l_s: _annotate_final(ax1, ev_s, ev_l_s, "Eval Loss={:.4f}")

    if tr_s: ax2.plot(tr_s, tr_p_s, label="Train PPL")
    if ev_s and any(v is not None for v in ev_p_s):
        y = [v if v is not None else np.nan for v in ev_p_s]
        ax2.plot(ev_s, y, label="Eval PPL")
    ax2.set_xlabel("Step"); ax2.set_ylabel("Perplexity"); ax2.legend(); ax2.grid(True, alpha=0.3)
    if ev_p_s and ev_p_s[-1] is not None:
        _annotate_final(ax2, ev_s, ev_p_s, "Eval PPL={:.2f}")

    out = RESULTS_DIR / f"executor_{name}_metrics.png"
    plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(out, dpi=120); plt.close()
    print(f"📊  Plots guardados en {out}")

def export_epoch_metrics(logs: List[Dict[str, Any]], out_json: Path):
    rows = []
    for r in logs:
        if "eval_loss" in r or "eval_exact_match" in r:
            ppl = r.get("eval_perplexity")
            if ppl is None and r.get("eval_loss") is not None:
                ppl = math.exp(min(float(r["eval_loss"]), 20))
            rows.append({
                "epoch": r.get("epoch"), "step": r.get("step"),
                "eval_loss": r.get("eval_loss"), "eval_perplexity": ppl,
                "eval_exact_match": r.get("eval_exact_match")
            })
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        last = rows[-1]
        print(f"🧾  Métricas por época: {out_json} "
              f"(última: loss={last.get('eval_loss')}, ppl={last.get('eval_perplexity')}, exact={last.get('eval_exact_match')})")
    else:
        print(f"⚠️  No se encontraron entradas de evaluación para exportar en {out_json}")

# ───────── Reconstrucción de prefijo ─────────
def _decode_text(tok, ids: List[int]) -> str:
    ids = list(ids)
    while ids and ids[-1] == tok.pad_token_id: ids.pop()
    return tok.decode(ids, skip_special_tokens=True)

def _reconstruct_prefix_and_answer(entry: Dict[str, Any], tok) -> Tuple[List[int], List[int], str, str]:
    input_ids: List[int] = list(entry["input_ids"])
    labels:    List[int] = list(entry["labels"])
    first_ans_idx = next((i for i, t in enumerate(labels) if t != -100), len(labels))
    prefix_ids = input_ids[:first_ans_idx]
    target_answer_ids = [t for t in labels if t != -100]
    if target_answer_ids and target_answer_ids[-1] == tok.eos_token_id:
        target_answer_ids = target_answer_ids[:-1]
    prompt_text = _decode_text(tok, prefix_ids.copy())
    target_answer_text = tok.decode(target_answer_ids, skip_special_tokens=True).strip()
    return prefix_ids, target_answer_ids, prompt_text, target_answer_text

# ───────── Logits Processor (solo dígitos) ─────────
_ALLOWED_PATTERN = re.compile(r"^[0-9\-\./\s]+$")

def _compute_allowed_token_ids(tok: AutoTokenizer, extra_allowed: Optional[Set[int]] = None) -> Set[int]:
    allowed: Set[int] = set()
    vocab = tok.get_vocab()
    inv_vocab = {i: t for t, i in vocab.items()}
    for i in range(len(inv_vocab)):
        t = inv_vocab.get(i)
        if t is None: continue
        if tok.special_tokens_map and t in tok.special_tokens_map.values():
            continue
        normalized = t.replace('▁', ' ')
        if _ALLOWED_PATTERN.match(normalized):
            allowed.add(i)
    if tok.eos_token_id is not None: allowed.add(tok.eos_token_id)
    if tok.pad_token_id is not None: allowed.add(tok.pad_token_id)
    for sym in ['\n', '\r']:
        ids = tok.encode(sym, add_special_tokens=False)
        for _id in ids: allowed.add(_id)
    if extra_allowed:
        allowed |= set(extra_allowed)
    return allowed

class AllowedTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids: Set[int]):
        self.allowed = set(int(i) for i in allowed_token_ids)
        self._last_device = None
        self._mask = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if (self._mask is None) or (self._last_device != scores.device) or (self._mask.size(-1) != scores.size(-1)):
            vocab_size = scores.size(-1)
            mask = torch.ones(vocab_size, dtype=torch.bool, device=scores.device)
            allowed_idx = torch.tensor(sorted(self.allowed), dtype=torch.long, device=scores.device)
            allowed_idx = allowed_idx[(allowed_idx >= 0) & (allowed_idx < vocab_size)]
            mask[allowed_idx] = False
            self._mask = mask
            self._last_device = scores.device
        scores = scores.masked_fill(self._mask, -float('inf'))
        return scores

# ───────── Trainer custom: genera desde prefijo ─────────
class CausalGenTrainer(Seq2SeqTrainer):
    def __init__(self, *args, logits_processor: Optional[LogitsProcessorList] = None,
                 gen_kwargs: Optional[Dict[str, Any]] = None, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._logits_processor = logits_processor
        self._gen_kwargs = gen_kwargs or {}
        self._tok = tokenizer

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Eval real:
        - calcula loss normal (si hay labels)
        - genera outputs DESDE EL PREFIJO (labels=-100 aún) para medir EM real
        """
        model.eval()
        has_labels = "labels" in inputs and inputs["labels"] is not None
        with torch.no_grad():
            loss = None
            if has_labels:
                outputs = model(**inputs)
                loss = outputs.loss

            # reconstruir prefijos por item y generar
            input_ids = inputs["input_ids"]
            labels = inputs.get("labels", None)

            device = next(model.parameters()).device
            batch_preds = []
            for b in range(input_ids.size(0)):
                ids = input_ids[b].tolist()
                lbs = labels[b].tolist() if labels is not None else [-100]*len(ids)

                # prefijo = hasta primer label != -100
                try:
                    first_ans_idx = next(i for i,t in enumerate(lbs) if t != -100)
                except StopIteration:
                    first_ans_idx = len(ids)
                prefix_ids = ids[:first_ans_idx]
                if len(prefix_ids) == 0:
                    prefix_ids = [self._tok.eos_token_id]

                inp = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0)
                attn = torch.ones_like(inp)  # ← máscara explícita para evitar warning y ambigüedad

                gen = model.generate(
                    input_ids=inp,
                    attention_mask=attn,  
                    pad_token_id=self._tok.eos_token_id,
                    eos_token_id=self._tok.eos_token_id,
                    logits_processor=self._logits_processor,
                    **self._gen_kwargs,
                )
                # SOLO la continuación generada
                gen_cont = gen[0][inp.size(1):].detach().cpu()
                batch_preds.append(gen_cont)

            # pad a misma longitud para devolver tensor 2D
            maxlen = max(t.size(0) for t in batch_preds) if batch_preds else 0
            if maxlen == 0:
                preds = torch.zeros((input_ids.size(0), 0), dtype=torch.long)
            else:
                preds = torch.full((len(batch_preds), maxlen), fill_value=self._tok.pad_token_id, dtype=torch.long)
                for i, t in enumerate(batch_preds):
                    preds[i, :t.size(0)] = t

            if prediction_loss_only:
                return (loss, None, None)

            labels_out = inputs.get("labels")
            return (loss, preds, labels_out)

# ───────── Muestras cualitativas ─────────
def sample_qualitative_predictions(model, tok, eval_ds: Dataset, out_json: Path, num_samples: int = 20,
                                   logits_processor: Optional[LogitsProcessorList] = None, max_new_tokens: int = 24):
    if len(eval_ds) == 0:
        print("⚠️  eval_ds vacío, no se generan muestras cualitativas."); return
    idxs = random.sample(range(len(eval_ds)), k=min(num_samples, len(eval_ds)))
    model.eval(); samples = []; device = next(model.parameters()).device
    for i in idxs:
        entry = {k: eval_ds[k][i] for k in ["input_ids", "attention_mask", "labels"]}
        prefix_ids, _, prompt_text, expected_answer = _reconstruct_prefix_and_answer(entry, tok)
        inp = torch.tensor(prefix_ids, dtype=torch.long).unsqueeze(0).to(device)
        attn = torch.ones_like(inp)
        with torch.no_grad():
            gen = model.generate(
                input_ids=inp, attention_mask=attn,
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
                logits_processor=logits_processor,
            )
        gen_text = tok.decode(gen[0][inp.size(1):], skip_special_tokens=True).strip()
        pred_can = canonify_text_answer(gen_text); exp_can = canonify_text_answer(expected_answer)
        ctx = prompt_text.splitlines()[0].strip() if prompt_text else ""
        samples.append({"context": ctx, "prompt": prompt_text,
                        "expected_answer": exp_can, "predicted_answer": pred_can})
    out_json.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"🧪  Muestras cualitativas guardadas en {out_json}")

# ───────── compute_metrics: usa predicciones generadas ─────────
def compute_metrics(eval_pred: EvalPrediction, tokenizer: AutoTokenizer):
    preds = eval_pred.predictions  # [bsz, Tgen] (continuación ya)
    labels = eval_pred.label_ids
    tok = tokenizer

    # EM real con ancla y canonificación
    bsz = preds.shape[0] if isinstance(preds, np.ndarray) else len(preds)
    exact_ok = 0
    for i in range(bsz):
        # gold
        gold_ids = [int(t) for t in labels[i] if t != -100]
        gold_txt = tok.decode(gold_ids, skip_special_tokens=True)
        gold_can = canonify_text_answer(gold_txt)

        # pred: convertir a lista antes de decode (evita OverflowError)
        pred_arr = preds[i]
        pred_ids = pred_arr.tolist() if hasattr(pred_arr, "tolist") else list(pred_arr)
        pred_txt = tok.decode(pred_ids, skip_special_tokens=True)
        pred_txt = _extract_after_answer_marker(pred_txt)
        pred_can = canonify_text_answer(pred_txt)

        if gold_can and pred_can and gold_can == pred_can:
            exact_ok += 1

    metrics = {"eval_exact_match": exact_ok / max(1, bsz)}
    return metrics

# ───────── Validación manual ─────────
def manual_eval_em(model, tok, eval_ds: Dataset, k: int = 64,
                   logits_processor: Optional[LogitsProcessorList] = None,
                   max_new_tokens: int = 24) -> Dict[str, Any]:
    if eval_ds is None or len(eval_ds) == 0:
        return {"n": 0, "ok": 0, "em": None}
    idxs = random.sample(range(len(eval_ds)), k=min(k, len(eval_ds)))
    device = next(model.parameters()).device
    ok = 0; rows = []
    model.eval()
    for i in idxs:
        entry = {k: eval_ds[k][i] for k in ["input_ids", "attention_mask", "labels"]}
        prefix_ids, _, prompt_text, expected_answer = _reconstruct_prefix_and_answer(entry, tok)
        inp = torch.tensor(prefix_ids, dtype=torch.long).unsqueeze(0).to(device)
        attn = torch.ones_like(inp)
        with torch.no_grad():
            gen = model.generate(
                input_ids=inp, attention_mask=attn,
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
                logits_processor=logits_processor,
            )
        gen_txt = tok.decode(gen[0][inp.size(1):], skip_special_tokens=True).strip()
        pred_can = canonify_text_answer(gen_txt)
        gold_can = canonify_text_answer(expected_answer)
        ok += int(pred_can == gold_can and bool(gold_can))
        rows.append({"prompt": prompt_text, "gold": gold_can, "pred": pred_can, "match": (pred_can == gold_can)})

    return {"n": len(idxs), "ok": ok, "em": (ok / max(1, len(idxs))), "samples": rows[:20]}

# ───────── Entrenamiento ─────────
def train_executor(name: str, token: str, base: str,
                   epochs: int, bs: int, accum: int, lr: float,
                   patience: int = 3, gen_new_tokens: int = 24):
    print(f"\n▶️  Entrenando executor «{name}» (EM real)")
    model, tok = load_llama_safe(base, token)

    # datasets
    def load_json(p: Path) -> List[Dict[str, Any]]:
        if not p.exists():
            print(f"⚠️  {p} no existe."); return []
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, list): return []
            return data
        except Exception as e:
            print(f"⚠️  Error leyendo {p}: {e}"); return []

    train_list = load_json(EXECUTORS_DIR / f"dataset_{name}_train.json")
    eval_list  = load_json(EXECUTORS_DIR / f"dataset_{name}_test.json")

    if len(train_list) == 0:
        print(f"⏭️  Sin datos de entrenamiento para «{name}». Ejecuta gen_executors_datasets_multistage.py. Saltando.")
        return

    train_ds = Dataset.from_list(train_list)
    eval_ds  = Dataset.from_list(eval_list) if len(eval_list) > 0 else None

    # QLoRA
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.config.use_cache = False
    try: model.config.pretraining_tp = 1
    except Exception: pass

    fp16, bf16 = detect_precision()
    steps_ep   = max(1, len(train_ds) // (bs * max(1, accum)))
    out_dir = Path(f"./models/executors/{name}"); out_dir.mkdir(parents=True, exist_ok=True)

    # logits processor solo para simple
    logits_processor = None
    if name == "simple":
        allowed_ids = _compute_allowed_token_ids(tok)
        logits_processor = LogitsProcessorList([AllowedTokensLogitsProcessor(allowed_ids)])

    gen_kwargs = {
        "max_new_tokens": int(gen_new_tokens),
        "do_sample": False, "num_beams": 1,
    }

    has_eval = eval_ds is not None and len(eval_ds) > 0
    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir), overwrite_output_dir=True,
        per_device_train_batch_size=bs, per_device_eval_batch_size=bs,
        gradient_accumulation_steps=accum, num_train_epochs=epochs,
        learning_rate=lr, lr_scheduler_type="cosine", warmup_ratio=0.1, weight_decay=0.0,
        eval_strategy=("epoch" if has_eval else "no"),
        save_strategy=("epoch" if has_eval else "steps"),
        load_best_model_at_end=has_eval, metric_for_best_model=("eval_exact_match" if has_eval else None),
        greater_is_better=True, save_total_limit=2,
        logging_strategy="steps", logging_steps=max(1, steps_ep // 5), logging_first_step=True,
        report_to="none",
        fp16=fp16, bf16=bf16, gradient_checkpointing=True, optim="paged_adamw_8bit",
        remove_unused_columns=False, dataloader_num_workers=2, dataloader_pin_memory=False,
        save_safetensors=True, group_by_length=False,
        predict_with_generate=True, generation_num_beams=1,
    )

    callbacks = [TrainPerplexityCallback()]
    if has_eval:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer = CausalGenTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=(eval_ds if has_eval else None),
        tokenizer=tok, data_collator=default_data_collator,
        compute_metrics=(lambda p: compute_metrics(p, tok)) if has_eval else None,
        callbacks=callbacks,
        logits_processor=logits_processor, gen_kwargs=gen_kwargs,
    )

    trainer.train()

    save_dir = out_dir / "trained_model"; model.save_pretrained(save_dir); tok.save_pretrained(save_dir)
    print(f"✅  Modelo «{name}» guardado en {save_dir}")

    plot_logs(trainer.state.log_history, name, smooth_w=0.6)
    metrics_json = out_dir / f"executor_{name}_metrics.json"
    export_epoch_metrics(trainer.state.log_history, metrics_json)

    # Muestras cualitativas + validación manual
    if has_eval:
        samples_json = out_dir / f"executor_{name}_samples.json"
        sample_qualitative_predictions(model, tok, eval_ds, samples_json, num_samples=20,
                                       logits_processor=logits_processor, max_new_tokens=gen_new_tokens)

        manual = manual_eval_em(model, tok, eval_ds, k=64,
                                logits_processor=logits_processor, max_new_tokens=gen_new_tokens)
        manual_json = out_dir / f"executor_{name}_manual_eval.json"
        manual_json.write_text(json.dumps(manual, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"🧪  Validación manual: EM={manual.get('em')} ({manual.get('ok')}/{manual.get('n')}) → {manual_json}")

    del trainer, model
    try: torch.cuda.empty_cache()
    except Exception: pass
    gc.collect()

# ───────── CLI ─────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser("Train executors (simple/complex) con QLoRA 4-bit — EM real")
    ap.add_argument("--hf_token",   required=True)
    ap.add_argument("--base_model", default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--epochs",     type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr",         type=float, default=1e-4)
    ap.add_argument("--patience",   type=int, default=3, help="Early stopping patience (épocas)")
    ap.add_argument("--gen_new_tokens", type=int, default=24, help="Tokens nuevos al generar en eval/predict")
    args = ap.parse_args()

    for task in ("simple", "complex"):
        train_executor(task, args.hf_token, args.base_model,
                       args.epochs, args.batch_size, args.grad_accum, args.lr,
                       patience=args.patience, gen_new_tokens=args.gen_new_tokens)
