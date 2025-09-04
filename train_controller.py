#!/usr/bin/env python
# coding: utf-8
"""
train_controller.py — Fine-tuning del controlador (LLaMA-2 chat + LoRA 4-bit)

NOTAS:
- Guarda trainer.state.log_history → controller_training_log.json
- Figura training_metrics.png (eval_loss, context_acc, route_acc)
- Matrices de confusión sobre el split de validación:
    • confusion_route.json  (SIMPLE vs COMPLEX)
    • confusion_context.json (C1..C12)
- Métricas finales del mejor modelo → best_model_metrics.json
- Resumen final impreso en consola (listo para pegar en el TFG)
"""

from __future__ import annotations
import argparse, gc, json, random, re, sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    default_data_collator,
    EarlyStoppingCallback,
)

# ─────────────────────────── CLI ───────────────────────────
cli = argparse.ArgumentParser()
cli.add_argument("--data", type=Path, required=True,
                 help="Ruta a controller_dataset.jsonl")
cli.add_argument("--model", type=str,
                 default="meta-llama/Llama-2-7b-chat-hf")
cli.add_argument("--out", type=Path, default=Path("./models/controller"))
cli.add_argument("--epochs", type=int, default=5)
cli.add_argument("--bsz", type=int, default=4)
cli.add_argument("--grad_accum", type=int, default=4)
cli.add_argument("--maxlen", type=int, default=512)
args = cli.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

# ─────────────────────────── Seeds ─────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ─────────────────────── Dataset & tokenizer ───────────────
raw = load_dataset("json", data_files=str(args.data), split="train")
raw = raw.train_test_split(test_size=0.1, seed=SEED)

tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
tok.pad_token = tok.eos_token

# ─────────────────── Regex para métricas/parse ─────────────
CTX_RE   = re.compile(r"\[CONTEXT_ID=(C\d+)\]")
ROUTE_RE = re.compile(r"\[ROUTE=(SIMPLE|COMPLEX)\]")

def _extract(txt: str):
    """Devuelve (context_id, route) a partir del texto generado/esperado."""
    ctx_m = CTX_RE.search(txt)
    ctx = ctx_m.group(1) if ctx_m else ""
    rt_m = ROUTE_RE.search(txt)
    route = rt_m.group(1) if rt_m else ""
    return ctx, route

# ────────────────────────── Tokenización ───────────────────
def encode(row):
    ids  = tok(row["input"] + "\n", add_special_tokens=False)["input_ids"]
    lbls = tok(row["output"],       add_special_tokens=False)["input_ids"]
    input_ids = (ids + lbls)[:args.maxlen]
    labels    = ([-100]*len(ids) + lbls)[:args.maxlen]
    pad = args.maxlen - len(input_ids)
    input_ids += [tok.pad_token_id] * pad
    labels    += [-100] * pad
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * args.maxlen,
        "labels": labels
    }

encoded = raw.map(encode, remove_columns=raw["train"].column_names)
train_ds, eval_ds = encoded["train"], encoded["test"]

# ─────────────────── Modelo 4-bit + LoRA ───────────────────
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base = AutoModelForCausalLM.from_pretrained(
    args.model, device_map="auto", quantization_config=bnb_cfg
)
base.gradient_checkpointing_enable()
base = prepare_model_for_kbit_training(base)

lora_cfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(base, lora_cfg)
model.config.use_cache = False

# ──────────────── Métricas personalizadas (eval) ───────────
def compute_metrics(eval_pred):
    """Calcula context_acc y route_acc a partir del decodificado."""
    preds_raw, label_ids = eval_pred
    if isinstance(preds_raw, tuple):  # compat
        preds_raw = preds_raw[0]

    # Greedy sobre logits token-a-token (coherente con el entrenamiento LM)
    pred_ids  = np.argmax(preds_raw, axis=-1)
    label_ids = np.where(label_ids == -100, tok.pad_token_id, label_ids)

    preds = tok.batch_decode(pred_ids, skip_special_tokens=True)
    refs  = tok.batch_decode(label_ids, skip_special_tokens=True)

    ctx_acc   = np.mean([_extract(p)[0] == _extract(r)[0] for p, r in zip(preds, refs)])
    route_acc = np.mean([_extract(p)[1] == _extract(r)[1] for p, r in zip(preds, refs)])
    return {"context_acc": ctx_acc, "route_acc": route_acc}

# ─────────────────────── Entrenamiento ─────────────────────
training_args = TrainingArguments(
    output_dir=str(args.out),
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.bsz,
    per_device_eval_batch_size=args.bsz,
    gradient_accumulation_steps=args.grad_accum,
    learning_rate=1e-4,
    warmup_ratio=0.1,
    bf16=True,
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
    greater_is_better=True,
    metric_for_best_model="route_acc",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tok,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print("🚀  Entrenando controlador...")
trainer.train()

# ─────────────────── Guardado de pesos ─────────────────────
out_final = args.out / "final"
model.save_pretrained(out_final)
tok.save_pretrained(out_final)
print(f"✅  Modelo guardado en {out_final}")

# ──────────────── LOGS → controller_training_log.json ───────
log_hist = trainer.state.log_history  # incluye train/eval por paso/época
log_path = args.out / "controller_training_log.json"
log_path.write_text(json.dumps(log_hist, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"📝  Log de entrenamiento: {log_path}")

# ─────────── Figura: eval_loss + context_acc + route_acc ────
# filtramos entradas con métricas de evaluación por época
hist_eval = [r for r in log_hist if "eval_loss" in r]
epochs    = [r.get("epoch") for r in hist_eval]
eval_loss = [r.get("eval_loss") for r in hist_eval]
ctx_acc   = [r.get("eval_context_acc") for r in hist_eval]
route_acc = [r.get("eval_route_acc") for r in hist_eval]

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(epochs, eval_loss, "o-", label="Eval loss")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(epochs, ctx_acc,   "s-", label="Context acc")
ax2.plot(epochs, route_acc, "x-", label="Route acc")
ax2.set_ylabel("Accuracy")

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper right")
plt.title("Controller fine-tuning — Eval loss / Accuracies")
plt.tight_layout()
fig_path = args.out / "training_metrics.png"
plt.savefig(fig_path, dpi=120); plt.close()
print(f"📈  Gráfica guardada en: {fig_path}")

# ─────────────── Predicción en validación y confusiones ─────
print("🔎  Generando predicciones sobre validación para matrices de confusión…")
pred_out = trainer.predict(eval_ds)
pred_ids = np.argmax(pred_out.predictions, axis=-1)
lbl_ids  = np.where(pred_out.label_ids == -100, tok.pad_token_id, pred_out.label_ids)

pred_txt = tok.batch_decode(pred_ids, skip_special_tokens=True)
ref_txt  = tok.batch_decode(lbl_ids,  skip_special_tokens=True)

pred_ctx, pred_route = zip(*[_extract(t) for t in pred_txt])
ref_ctx,  ref_route  = zip(*[_extract(t) for t in ref_txt])

# Matriz de confusión (route)
labels_route = ["SIMPLE", "COMPLEX"]
idx_route = {lab:i for i, lab in enumerate(labels_route)}
cm_route = [[0,0],[0,0]]
for pr, rr in zip(pred_route, ref_route):
    if pr not in idx_route or rr not in idx_route:
        continue
    cm_route[idx_route[rr]][idx_route[pr]] += 1

# Matriz de confusión (context) dinámica
ctx_labels = sorted({c for c in ref_ctx if c})
idx_ctx = {lab:i for i, lab in enumerate(ctx_labels)}
cm_ctx = [[0 for _ in ctx_labels] for _ in ctx_labels]
for pc, rc in zip(pred_ctx, ref_ctx):
    if rc in idx_ctx and pc in idx_ctx:
        cm_ctx[idx_ctx[rc]][idx_ctx[pc]] += 1

conf_route_path = args.out / "confusion_route.json"
conf_ctx_path   = args.out / "confusion_context.json"
(conf_route_path).write_text(json.dumps({
    "labels": labels_route,
    "matrix": cm_route
}, indent=2), encoding="utf-8")
(conf_ctx_path).write_text(json.dumps({
    "labels": ctx_labels,
    "matrix": cm_ctx
}, indent=2), encoding="utf-8")
print(f"🧭  Matriz de confusión (route): {conf_route_path}")
print(f"🧩  Matriz de confusión (context): {conf_ctx_path}")

# ─────────────── Métricas finales del mejor modelo ──────────
best_metrics = {
    "best_model_checkpoint": trainer.state.best_model_checkpoint,
    "best_metric":           trainer.state.best_metric,
    # Tomamos la última evaluación registrada (suele corresponder al mejor si load_best_model_at_end=True)
    "final_eval": {
        "epoch":       epochs[-1] if epochs else None,
        "eval_loss":   eval_loss[-1] if eval_loss else None,
        "context_acc": ctx_acc[-1] if ctx_acc else None,
        "route_acc":   route_acc[-1] if route_acc else None,
    }
}
best_path = args.out / "best_model_metrics.json"
best_path.write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
print(f"🏁  best_model_metrics.json guardado en: {best_path}")

# ─────────────── Resumen para §6.2.1 (stdout) ───────────────
print("\n──────────── RESUMEN CONTROLADOR (VALIDACIÓN) ────────────")
if epochs:
    print(f"• Épocas: {int(round(epochs[-1]))}")
if eval_loss:
    print(f"• Eval loss (última): {eval_loss[-1]:.4f}")
if ctx_acc:
    print(f"• Context accuracy:   {ctx_acc[-1]*100:5.2f}%")
if route_acc:
    print(f"• Route accuracy:     {route_acc[-1]*100:5.2f}%")
print(f"• Checkpoint óptimo:  {trainer.state.best_model_checkpoint}")
print("• Ficheros generados:")
print(f"   - {log_path.name}")
print(f"   - {fig_path.name}")
print(f"   - {conf_route_path.name}")
print(f"   - {conf_ctx_path.name}")
print(f"   - {best_path.name}")
print("───────────────────────────────────────────────────────────")

# ────────────────────────── Limpieza ─────────────────────────
del trainer, model, train_ds, eval_ds
gc.collect(); 
try:
    torch.cuda.empty_cache()
except Exception:
    pass
sys.exit(0)
