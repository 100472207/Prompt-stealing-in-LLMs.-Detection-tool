#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_detector.py — Entrena y valida el detector de fugas (safety-first) + PRUEBAS DE FUNCIONAMIENTO

Entrena sobre:
  ./data/detector/train.jsonl
  ./data/detector/val.jsonl
  ./data/detector/test_ood.jsonl

Política safety-first:
  - Pérdida ponderada: --pos_class_weight (↑ para penalizar más FN)
  - Calibración de umbral por recall objetivo: --policy safety_first --target_recall 0.995 --max_fpr_cap 0.20
  - Zona gris = fuga: --uncertainty_margin (por defecto 0.02; se ensancha si el umbral queda ≥ 0.99)

Genera pruebas en <OUTPUT_DIR>:
  ├─ model/                               # pesos y tokenizer
  ├─ decision_threshold.json              # umbral calibrado + política
  ├─ metrics_val.json                     # métricas globales (val)
  ├─ metrics_ood.json                     # métricas globales (ood)
  ├─ metrics_by_leaktype_val.json         # TPR por tipo de fuga; FPR en negativos
  ├─ metrics_by_leaktype_ood.json
  ├─ metrics_by_filter_val.json           # TPR/FPR por filtro
  ├─ metrics_by_filter_ood.json
  ├─ threshold_curve_val.csv              # fpr,tpr,threshold
  ├─ pr_curve_val.csv                     # precision,recall,threshold
  ├─ roc_val.png, pr_val.png, roc_ood.png, pr_ood.png
  ├─ confusion_val.json, confusion_ood.json
  ├─ examples_val_errors.jsonl            # falsos + / falsos - (top K)
  ├─ examples_ood_errors.jsonl
  ├─ smoke_test.json
  ├─ trainer_log.json
  └─ report.json

Uso (ejemplo):
  python train_detector.py --model_name roberta-base --output_dir ./detector/leakage_classifier \
    --pos_class_weight 2.0 --policy safety_first --target_recall 0.995 --max_fpr_cap 0.20 \
    --uncertainty_margin 0.02 --fp16
"""

from __future__ import annotations
import argparse, json, os, random, csv, sys, platform
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np

from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_fscore_support,
    accuracy_score, confusion_matrix, roc_curve, precision_recall_curve,
    brier_score_loss, f1_score
)

import matplotlib.pyplot as plt
import torch
from torch import nn
from datasets import DatasetDict, Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, default_data_collator, set_seed
)

# ───────── Config ─────────
SEED = int(os.getenv("SEED", 42))
set_seed(SEED)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
DET_DIR  = DATA_DIR / "detector"


# ───────── Utilidades E/S ─────────
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rows.append(json.loads(line))
    return rows


def read_label_schema(det_dir: Path) -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    Lee ./data/detector/label_schema.json si existe para fijar id2label/label2id.
    Formato esperado: {"0": "no", "1": "yes"}
    """
    schema_p = det_dir / "label_schema.json"
    if not schema_p.exists():
        return {0: "no", 1: "yes"}, {"no": 0, "yes": 1}
    try:
        obj = json.loads(schema_p.read_text(encoding="utf-8"))
        # Normalizamos claves a int
        id2label = {int(k): str(v) for k, v in obj.items()}
        label2id = {v: k for k, v in id2label.items()}
        # Validación mínima
        if 0 not in id2label or 1 not in id2label:
            raise ValueError("label_schema.json debe mapear 0 y 1")
        return id2label, label2id
    except Exception as e:
        print(f"⚠️ No se pudo leer label_schema.json: {e} — usando {'no','yes'} por defecto")
        return {0: "no", 1: "yes"}, {"no": 0, "yes": 1}


def build_hf_dataset(train_p: Path, val_p: Path, ood_p: Path) -> Tuple[DatasetDict, Dict[str, List[Dict[str, Any]]]]:
    """
    Devuelve:
      - DatasetDict HF con columnas 'text' y 'label'
      - Metadatos por split (lista de dicts alineada por orden) con campos opcionales: 'filter', 'leak_type'
    """
    def to_hf_and_meta(rows):
        ds = Dataset.from_list([{"text": r["text"], "label": int(r["label"])} for r in rows])
        meta = []
        for r in rows:
            meta.append({
                "filter": r.get("filter", None),
                "leak_type": r.get("leak_type", None),
                "text": r.get("text", ""),
                "label": int(r.get("label", 0)),
            })
        return ds, meta

    train_rows = read_jsonl(train_p)
    val_rows   = read_jsonl(val_p)
    ood_rows   = read_jsonl(ood_p)

    train_ds, meta_train = to_hf_and_meta(train_rows)
    val_ds,   meta_val   = to_hf_and_meta(val_rows)
    ood_ds,   meta_ood   = to_hf_and_meta(ood_rows)

    dsd = DatasetDict({"train": train_ds, "validation": val_ds, "test_ood": ood_ds})
    metas = {"train": meta_train, "validation": meta_val, "test_ood": meta_ood}
    return dsd, metas


def make_tokenizer(name: str, max_len: int):
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    def encode(batch):
        enc = tok(
            batch["text"],
            max_length=max_len,
            padding="max_length",
            truncation=True,
        )
        enc["labels"] = batch["label"]
        return enc

    return tok, encode


def compute_metrics_builder():
    def compute_metrics(eval_pred):
        # Soporta EvalPrediction (objeto) y tupla (preds, labels)
        if hasattr(eval_pred, "predictions"):
            logits = eval_pred.predictions
            labels = eval_pred.label_ids
        else:
            logits, labels = eval_pred

        if isinstance(logits, list):
            logits = np.asarray(logits)
        if hasattr(logits, "ndim") and logits.ndim == 1:
            # Caso modelos que devuelven logit escalar (pos)
            probs = 1 / (1 + np.exp(-logits.reshape(-1)))
        else:
            probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]

        labels = labels.astype(int)
        preds = (probs >= 0.5).astype(int)  # logging preliminar con 0.5
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        acc = accuracy_score(labels, preds)
        try:
            auroc = roc_auc_score(labels, probs)
        except Exception:
            auroc = float("nan")
        try:
            auprc = average_precision_score(labels, probs)
        except Exception:
            auprc = float("nan")
        try:
            brier = brier_score_loss(labels, probs)
        except Exception:
            brier = float("nan")
        return {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "auroc": auroc, "auprc": auprc, "brier": brier}
    return compute_metrics


# ───────── Curvas y evaluación ─────────
def plot_curves(probs: np.ndarray, labels: np.ndarray, out_dir: Path, split: str):
    try:
        fpr, tpr, thr_roc = roc_curve(labels, probs)
        auroc = roc_auc_score(labels, probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={auroc:.3f}")
        plt.plot([0,1],[0,1], linestyle="--")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title(f"ROC {split}")
        plt.legend()
        plt.savefig(out_dir / f"roc_{split}.png", dpi=140, bbox_inches="tight"); plt.close()

        precisions, recalls, thresholds = precision_recall_curve(labels, probs)
        auprc = average_precision_score(labels, probs)
        plt.figure()
        plt.plot(recalls, precisions, label=f"AUC={auprc:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR {split}")
        plt.legend()
        plt.savefig(out_dir / f"pr_{split}.png", dpi=140, bbox_inches="tight"); plt.close()

        # Persistimos curvas
        with open(out_dir / f"threshold_curve_{split}.csv", "w", newline="") as fw:
            w = csv.writer(fw); w.writerow(["fpr","tpr","threshold"])
            L = min(len(fpr), len(tpr), len(thr_roc))
            for i in range(L):
                w.writerow([float(fpr[i]), float(tpr[i]), float(thr_roc[i])])

        with open(out_dir / f"pr_curve_{split}.csv", "w", newline="") as fw:
            w = csv.writer(fw); w.writerow(["precision","recall","threshold"])
            for i, th in enumerate(thresholds):
                w.writerow([float(precisions[i]), float(recalls[i]), float(th)])
    except Exception as e:
        print(f"⚠️ No se pudieron generar curvas/tablas para {split}: {e}")


# ───────── Decisión con política (zona gris = fuga) ─────────
def predict_with_policy(probs: np.ndarray, thr: float, margin: float = 0.0) -> np.ndarray:
    if margin <= 0:
        return (probs >= thr).astype(int)
    lo = max(0.0, thr - margin)
    preds = np.zeros_like(probs, dtype=int)
    preds[probs >= thr] = 1
    # “zona gris” [lo, thr): por seguridad, también 1
    mask_gray = (probs >= lo) & (probs < thr)
    preds[mask_gray] = 1
    return preds


def calibrate_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    policy: str = "safety_first",
    target_fpr: float = 0.01,
    target_recall: float = 0.995,
    max_fpr_cap: float = 0.20
) -> float:
    """
    precision_first: umbral por FPR objetivo.
    safety_first: elegir el UMBRAL MÁS BAJO que consiga recall >= target_recall,
                  prefiriendo el de menor FPR y respetando max_fpr_cap si es posible.
    """
    fpr, tpr, thr_roc = roc_curve(labels, probs)
    precisions, recalls, thr_pr = precision_recall_curve(labels, probs)

    if policy == "precision_first":
        if len(thr_roc) == 0:
            return 0.5
        idx = int(np.argmin(np.abs(fpr - target_fpr)))
        return float(thr_roc[idx])

    # safety_first
    if len(thr_pr) == 0:
        return 0.5

    candidates = []
    for r, p, th in zip(recalls[:-1], precisions[:-1], thr_pr):
        if r >= target_recall:
            preds = (probs >= th).astype(int)
            tn = int(((labels == 0) & (preds == 0)).sum())
            fp = int(((labels == 0) & (preds == 1)).sum())
            fpr_here = fp / max(1, (tn + fp))
            candidates.append((th, r, fpr_here))

    if not candidates:
        # Fallback: mejor F1
        best, best_f1 = 0.5, -1.0
        for th in thr_pr:
            f1 = f1_score(labels, (probs >= th).astype(int))
            if f1 > best_f1:
                best_f1, best = f1, th
        return float(best)

    # Preferimos FPR mínimo (≤ cap si es posible); desempate por threshold más bajo
    candidates.sort(key=lambda x: (x[2], x[0]))
    for th, _, f in candidates:
        if f <= max_fpr_cap:
            return float(th)
    return float(candidates[0][0])


def eval_with_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    thr: float,
    margin: float = 0.0
) -> Dict[str, Any]:
    preds = predict_with_policy(probs, thr, margin)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds).tolist()
    try:
        auroc = roc_auc_score(labels, probs)
        auprc = average_precision_score(labels, probs)
        brier = brier_score_loss(labels, probs)
    except Exception:
        auroc = float("nan"); auprc = float("nan"); brier = float("nan")
    return {
        "threshold": float(thr),
        "uncertainty_margin": float(margin),
        "accuracy": acc, "precision": p, "recall": r, "f1": f1,
        "auroc": auroc, "auprc": auprc, "brier": brier,
        "confusion_matrix": cm
    }


# ───────── Métricas por grupo (leak_type / filtro) ─────────
def group_metrics_by_leaktype(
    probs: np.ndarray, labels: np.ndarray, metas: List[Dict[str, Any]], thr: float, margin: float
) -> Dict[str, Any]:
    leak = [m.get("leak_type", "none") or "none" for m in metas]
    leak_set = sorted(set(leak))
    preds = predict_with_policy(probs, thr, margin)
    out: Dict[str, Any] = {}

    for t in leak_set:
        idx = [i for i, lt in enumerate(leak) if lt == t]
        if not idx:
            continue
        y = labels[idx]
        yhat = preds[idx]
        support = int(len(idx))
        pos = (y == 1)
        neg = (y == 0)
        tpr = float(yhat[pos].mean()) if pos.any() else None   # recall en positivos
        fpr = float(yhat[neg].mean()) if neg.any() else None   # tasa de falsos pos en negativos
        out[t] = {"support": support, "tpr_if_pos": tpr, "fpr_if_neg": fpr}
    return out


def group_metrics_by_filter(
    probs: np.ndarray, labels: np.ndarray, metas: List[Dict[str, Any]], thr: float, margin: float
) -> Dict[str, Any]:
    filt = [m.get("filter", "unknown") or "unknown" for m in metas]
    filt_set = sorted(set(filt))
    preds = predict_with_policy(probs, thr, margin)
    out: Dict[str, Any] = {}
    for f in filt_set:
        idx = [i for i, fx in enumerate(filt) if fx == f]
        if not idx:
            continue
        y = labels[idx]; yhat = preds[idx]
        pos = (y == 1); neg = (y == 0)
        entry = {
            "n": int(len(idx)),
            "positives": int(pos.sum()),
            "negatives": int(neg.sum()),
            "tpr": float(yhat[pos].mean()) if pos.any() else None,
            "fpr": float(yhat[neg].mean()) if neg.any() else None,
        }
        out[f] = entry
    return out


# ───────── Guardado de ejemplos de error ─────────
def dump_error_examples(
    out_path: Path, metas: List[Dict[str, Any]], probs: np.ndarray, labels: np.ndarray, thr: float, margin: float, top_k: int = 200
):
    preds = predict_with_policy(probs, thr, margin)
    rows = []
    for i in range(len(labels)):
        if preds[i] != labels[i]:
            rows.append({
                "pred": int(preds[i]),
                "label": int(labels[i]),
                "prob": float(probs[i]),
                "filter": metas[i].get("filter"),
                "leak_type": metas[i].get("leak_type", "none"),
                "text": metas[i].get("text", "")[:800]
            })
    # Orden correcto: FN primero (prob ascendente), luego FP (prob descendente)
    def _rank(r):
        is_fn = (r["label"] == 1 and r["pred"] == 0)
        group = 0 if is_fn else 1
        score = r["prob"] if is_fn else -r["prob"]
        return (group, score)
    rows = sorted(rows, key=_rank)

    with out_path.open("w", encoding="utf-8") as fw:
        for r in rows[:top_k]:
            fw.write(json.dumps(r, ensure_ascii=False) + "\n")


# ───────── Smoke test ─────────
def run_smoke_test(
    probs: np.ndarray, labels: np.ndarray, thr: float, metas: List[Dict[str, Any]], margin: float, n: int = 10
) -> Dict[str, Any]:
    idx_pos = [i for i, y in enumerate(labels) if y == 1]
    idx_neg = [i for i, y in enumerate(labels) if y == 0]
    random.shuffle(idx_pos); random.shuffle(idx_neg)
    idx_pos = idx_pos[:min(n, len(idx_pos))]
    idx_neg = idx_neg[:min(n, len(idx_neg))]
    preds = predict_with_policy(probs, thr, margin)

    pos_ok = int(sum(int(preds[i] == 1) for i in idx_pos))
    neg_ok = int(sum(int(preds[i] == 0) for i in idx_neg))
    total_ok = pos_ok + neg_ok
    total = len(idx_pos) + len(idx_neg)

    samples = []
    for i in idx_pos + idx_neg:
        samples.append({
            "label": int(labels[i]),
            "pred": int(preds[i]),
            "prob": float(probs[i]),
            "filter": metas[i].get("filter"),
            "leak_type": metas[i].get("leak_type", "none"),
            "snippet": metas[i].get("text", "")[:200]
        })
    return {
        "n_pos": len(idx_pos), "n_neg": len(idx_neg),
        "pos_ok": pos_ok, "neg_ok": neg_ok,
        "acc_subset": float(total_ok / max(1, total)),
        "threshold": float(thr),
        "uncertainty_margin": float(margin),
        "examples": samples
    }


# ───────── Trainer ponderado ─────────
class WeightedTrainer(Trainer):
    def __init__(self, *args, pos_class_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        # weight = [w_neg, w_pos]; por defecto w_neg=1.0
        self._class_weight = torch.tensor([1.0, float(pos_class_weight)], dtype=torch.float)

    # Acepta argumentos extra (e.g., num_items_in_batch) para compatibilidad con HF recientes
    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        labels = inputs.get("labels")
        inputs_no_labels = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**inputs_no_labels)
        logits = outputs.get("logits")
        cw = self._class_weight.to(logits.device)
        loss_fct = nn.CrossEntropyLoss(weight=cw)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# ───────── Main ─────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="roberta-base")
    ap.add_argument("--output_dir", default="./detector/leakage_classifier")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--fp16", action="store_true")

    # Política de calibración
    ap.add_argument("--policy", choices=["precision_first", "safety_first"], default="safety_first",
                    help="precision_first = FPR objetivo; safety_first = Recall objetivo (ultraconservador)")
    ap.add_argument("--target_fpr", type=float, default=0.01, help="FPR objetivo (solo precision_first)")
    ap.add_argument("--target_recall", type=float, default=0.995,
                    help="Recall objetivo en validación (safety_first)")
    ap.add_argument("--max_fpr_cap", type=float, default=0.20,
                    help="Tope de FPR aceptable cuando recall≥objetivo (safety_first)")
    ap.add_argument("--uncertainty_margin", type=float, default=0.02,
                    help="Banda gris alrededor del umbral en la que se decide fuga (≥lo y <thr)")

    # Pérdida ponderada
    ap.add_argument("--pos_class_weight", type=float, default=2.0,
                    help="Peso de la clase positiva en la pérdida (safety-first)")

    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Carga datasets
    train_p = DET_DIR / "train.jsonl"
    val_p   = DET_DIR / "val.jsonl"
    ood_p   = DET_DIR / "test_ood.jsonl"
    if not (train_p.exists() and val_p.exists() and ood_p.exists()):
        raise SystemExit("❌ Faltan JSONL del detector. Ejecuta primero generate_detector_dataset.py")

    dsd, metas = build_hf_dataset(train_p, val_p, ood_p)

    # Tokenizador y codificación
    tok, encode = make_tokenizer(args.model_name, args.max_length)
    dsd = dsd.map(encode, batched=True, remove_columns=["text", "label"])

    # Mapeo de etiquetas (desde label_schema.json si existe)
    id2label, label2id = read_label_schema(DET_DIR)

    # Modelo
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    # Entrenamiento
    steps_per_epoch = max(1, len(dsd["train"]) // args.batch_size)
    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",   # ← parámetro correcto en HF
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=max(1, steps_per_epoch // 5),
        report_to="none",
        fp16=args.fp16,
        save_total_limit=2,
        seed=SEED,
        data_seed=SEED,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=dsd["train"],
        eval_dataset=dsd["validation"],
        tokenizer=tok,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics_builder(),
        pos_class_weight=args.pos_class_weight,
    )

    trainer.train()
    (out_dir / "trainer_log.json").write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")

    # Guardar mejor modelo
    trainer.save_model(out_dir / "model")
    tok.save_pretrained(out_dir / "model")

    # ===== Predicciones para pruebas =====
    def get_probs(ds: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        preds = trainer.predict(ds)
        logits = preds.predictions
        if isinstance(logits, list):
            logits = np.asarray(logits)
        if hasattr(logits, "ndim") and logits.ndim == 1:
            probs = 1 / (1 + np.exp(-logits))
        else:
            probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
        return probs, preds.label_ids.astype(int)

    probs_val, y_val = get_probs(dsd["validation"])
    probs_ood, y_ood = get_probs(dsd["test_ood"])

    # Curvas + tablas reproducibles
    plot_curves(probs_val, y_val, out_dir, split="val")
    plot_curves(probs_ood, y_ood, out_dir, split="ood")

    # Calibración de umbral según política elegida
    thr = calibrate_threshold(
        probs_val, y_val,
        policy=args.policy,
        target_fpr=args.target_fpr,
        target_recall=args.target_recall,
        max_fpr_cap=args.max_fpr_cap
    )
    # Ensancha banda si el umbral queda demasiado alto
    if args.policy == "safety_first" and thr >= 0.99:
        args.uncertainty_margin = max(args.uncertainty_margin, 0.02)

    decision_info = {
        "threshold": float(thr),
        "policy": args.policy,
        "target_fpr": args.target_fpr,
        "target_recall": args.target_recall,
        "max_fpr_cap": args.max_fpr_cap,
        "uncertainty_margin": args.uncertainty_margin,
        "pos_class_weight": args.pos_class_weight,
    }
    (out_dir / "decision_threshold.json").write_text(json.dumps(decision_info, indent=2), encoding="utf-8")

    # Métricas globales (aplican política de decisión con margen)
    m_val = eval_with_threshold(probs_val, y_val, thr, margin=args.uncertainty_margin)
    m_ood = eval_with_threshold(probs_ood, y_ood, thr, margin=args.uncertainty_margin)
    (out_dir / "metrics_val.json").write_text(json.dumps(m_val, indent=2), encoding="utf-8")
    (out_dir / "metrics_ood.json").write_text(json.dumps(m_ood, indent=2), encoding="utf-8")
    (out_dir / "confusion_val.json").write_text(json.dumps({"cm": m_val["confusion_matrix"]}, indent=2), encoding="utf-8")
    (out_dir / "confusion_ood.json").write_text(json.dumps({"cm": m_ood["confusion_matrix"]}, indent=2), encoding="utf-8")

    # Métricas por leak_type / filtro
    leak_val = group_metrics_by_leaktype(probs_val, y_val, metas["validation"], thr, args.uncertainty_margin)
    leak_ood = group_metrics_by_leaktype(probs_ood, y_ood, metas["test_ood"],  thr, args.uncertainty_margin)
    (out_dir / "metrics_by_leaktype_val.json").write_text(json.dumps(leak_val, indent=2), encoding="utf-8")
    (out_dir / "metrics_by_leaktype_ood.json").write_text(json.dumps(leak_ood, indent=2), encoding="utf-8")

    filt_val = group_metrics_by_filter(probs_val, y_val, metas["validation"], thr, args.uncertainty_margin)
    filt_ood = group_metrics_by_filter(probs_ood, y_ood, metas["test_ood"],  thr, args.uncertainty_margin)
    (out_dir / "metrics_by_filter_val.json").write_text(json.dumps(filt_val, indent=2), encoding="utf-8")
    (out_dir / "metrics_by_filter_ood.json").write_text(json.dumps(filt_ood, indent=2), encoding="utf-8")

    # Banco de casos (errores) para inspección manual
    dump_error_examples(out_dir / "examples_val_errors.jsonl", metas["validation"], probs_val, y_val, thr, args.uncertainty_margin, top_k=300)
    dump_error_examples(out_dir / "examples_ood_errors.jsonl", metas["test_ood"],   probs_ood, y_ood, thr, args.uncertainty_margin, top_k=300)

    # Smoke test con ejemplos reales
    smoke_val = run_smoke_test(probs_val, y_val, thr, metas["validation"], args.uncertainty_margin, n=10)
    smoke_ood = run_smoke_test(probs_ood, y_ood, thr, metas["test_ood"], args.uncertainty_margin, n=10)
    (out_dir / "smoke_test.json").write_text(json.dumps({"val": smoke_val, "ood": smoke_ood}, indent=2), encoding="utf-8")

    # Informe integral
    report = {
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "model_name": args.model_name,
            "seed": SEED
        },
        "data_counts": {
            "train": len(dsd["train"]),
            "validation": len(dsd["validation"]),
            "test_ood": len(dsd["test_ood"]),
        },
        "threshold": decision_info,
        "global_metrics": {"val": m_val, "ood": m_ood},
        "by_leaktype": {"val": leak_val, "ood": leak_ood},
        "by_filter": {"val": filt_val, "ood": filt_ood},
        "smoke_test": {"val": smoke_val, "ood": smoke_ood},
        "artifacts": [
            "decision_threshold.json",
            "metrics_val.json", "metrics_ood.json",
            "metrics_by_leaktype_val.json", "metrics_by_leaktype_ood.json",
            "metrics_by_filter_val.json", "metrics_by_filter_ood.json",
            "roc_val.png", "pr_val.png", "roc_ood.png", "pr_ood.png",
            "threshold_curve_val.csv", "pr_curve_val.csv",
            "examples_val_errors.jsonl", "examples_ood_errors.jsonl",
            "smoke_test.json", "trainer_log.json"
        ]
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Resumen en consola
    print("\n=== Detector entrenado (safety-first) ===")
    print("Umbral calibrado:", thr, "| política:", args.policy,
          "| target_recall:", args.target_recall, "| max_fpr_cap:", args.max_fpr_cap,
          "| pos_class_weight:", args.pos_class_weight, "| uncertainty_margin:", args.uncertainty_margin)
    print("VAL  ->", json.dumps({k: report["global_metrics"]["val"][k] for k in ["accuracy","precision","recall","f1","auroc","auprc"]}, indent=2))
    print("OOD  ->", json.dumps({k: report["global_metrics"]["ood"][k] for k in ["accuracy","precision","recall","f1","auroc","auprc"]}, indent=2))
    print(f"✅ Artefactos y PRUEBAS guardados en: {out_dir}")

if __name__ == "__main__":
    main()
