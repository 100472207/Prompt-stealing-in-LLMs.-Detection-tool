# Prompt-stealing-in-LLMs.-Detection-tool
# Prompt Stealing en LLMs: Herramienta de Detección — Reproducibilidad

Este repositorio permite **replicar** los resultados del TFG _“Prompt Stealing en LLMs: Herramienta de detección”_ ejecutando, en orden, los siguientes scripts:

1. [`generate_controller_dataset.py`](./generate_controller_dataset.py)
2. [`train_controller.py`](./train_controller.py)
3. [`gen_executors_datasets.py`](./gen_executors_datasets.py)
4. [`train_executors.py`](./train_executors.py)
5. [`generate_detector_dataset.py`](./generate_detector_dataset.py)
6. [`train_detector.py`](./train_detector.py)

> El pipeline entrena un **controlador**, dos **ejecutores** (SIMPLE y COMPLEX) sobre **DeepMind Mathematics**, y un **detector** calibrado con política **safety_first**.

---

## Tabla de Contenidos

- [Requisitos del sistema](#requisitos-del-sistema)
- [Instalación](#instalación)
- [Estructura de carpetas](#estructura-de-carpetas)
- [Ejecución paso a paso](#ejecución-paso-a-paso)
  - [1) Dataset del controlador](#1-dataset-del-controlador-generate_controller_datasetpy)
  - [2) Entrenamiento del controlador](#2-entrenamiento-del-controlador-train_controllerpy)
  - [3) Datasets de los ejecutores](#3-datasets-de-los-ejecutores-gen_executors_datasetspy)
  - [4) Entrenamiento de ejecutores](#4-entrenamiento-de-ejecutores-train_executorspy)
  - [5) Dataset del detector](#5-dataset-del-detector-generate_detector_datasetpy)
  - [6) Entrenamiento del detector-safety-first](#6-entrenamiento-del-detector-safety-first-train_detectorpy)
- [Resultados esperados](#resultados-esperados)
- [Notas de reproducibilidad](#notas-de-reproducibilidad)
- [Citación](#citación)

---

## Requisitos del sistema

- **SO**: Linux x86_64 (recomendado Ubuntu 20.04+).
- **Python**: 3.10–3.11.
- **GPU**: NVIDIA con CUDA (≥ 12 GB VRAM recomendado; ideal ≥ 24 GB para QLoRA 4-bit).
- **CUDA/cuDNN**: versión compatible con la instalación de PyTorch.
- **Hugging Face**: cuenta con acceso a modelos LLaMA-2 y **token** activo.
- **Espacio en disco**: ≥ 15 GB (datasets + checkpoints).

---

## Instalación

```bash
# 1) Entorno virtual
python -m venv .venv
source .venv/bin/activate

# 2) PyTorch (elige la variante con CUDA que te corresponda)
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3) Dependencias principales
pip install -U "transformers>=4.40" "datasets>=2.19" "accelerate>=0.30" \
  "peft>=0.11" "bitsandbytes>=0.43" "scikit-learn>=1.4" \
  "matplotlib>=3.8" "numpy>=1.26" "num2words>=0.5"

# 4) Autenticación en Hugging Face (necesario para LLaMA-2)
huggingface-cli login
# Alternativa: exportar el token si un script lo requiere explícitamente
# export HF_TOKEN="hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXX"


Comprobación rápida:

python - << 'PY'
import torch, bitsandbytes as bnb; print("CUDA OK:", torch.cuda.is_available())
PY

Estructura de carpetas
.
├── generate_controller_dataset.py
├── train_controller.py
├── gen_executors_datasets.py
├── train_executors.py
├── generate_detector_dataset.py
├── train_detector.py
├── data/                         # (se crea automáticamente si no existe)
│   ├── controller/
│   ├── executors/
│   └── detector/
└── models/
    ├── controller/
    └── executors/
        ├── simple/
        └── complex/


Puedes cambiar la raíz de datos con DATA_DIR (p.ej., export DATA_DIR=./data). Por defecto es ./data.

Ejecución paso a paso

Recomendado exportar previamente:

export DATA_DIR=./data
export CUDA_VISIBLE_DEVICES=0     # ajusta según tu equipo

1) Dataset del controlador (generate_controller_dataset.py)

Propósito: Construye el dataset del controlador a partir de DeepMind Mathematics (descarga/caché automática) y genera estadísticas.

Entrada: no requiere ficheros locales; usa datasets para obtener el corpus.

Salida (por defecto):

./data/controller/controller_dataset.jsonl

./data/controller/controller_dataset_stats.json

CLI:

python ./generate_controller_dataset.py \
  --out-dir "$DATA_DIR/controller" \
  --target-per-filter 80 \
  --samples-per-category 5 \
  --seed 42 --timeout 90

2) Entrenamiento del controlador (train_controller.py)

Propósito: Fine-tuning del controlador (LLaMA-2-7b-chat + LoRA 4-bit).

Entrada:

./data/controller/controller_dataset.jsonl

Salida (por defecto en ./models/controller/):

controller_training_log.json, training_metrics.png

confusion_route.json, confusion_context.json

best_model_metrics.json

CLI:

python ./train_controller.py \
  --data "$DATA_DIR/controller/controller_dataset.jsonl" \
  --model meta-llama/Llama-2-7b-chat-hf \
  --out ./models/controller \
  --epochs 5 --bsz 4 --grad_accum 4 --maxlen 512


Asegúrate de estar autenticado en Hugging Face para descargar el modelo base.

3) Datasets de los ejecutores (gen_executors_datasets.py)

Propósito: Genera datasets SIMPLE y COMPLEX (splits train/test), con auditoría y normalización específica.

Entrada: el script usa el corpus matemático y requiere token HF válido.

Salidas (por defecto en ./data/executors/):

dataset_simple_train.json, dataset_simple_test.json

dataset_complex_train.json, dataset_complex_test.json

executors_dataset_stats.json, executors_dataset_audit_simple.json, executors_dataset_audit_complex.json

CLI (parámetros principales):

# Nota: --hf_token es obligatorio
python ./gen_executors_datasets.py \
  --hf_token "$HF_TOKEN" \
  --base_model meta-llama/Llama-2-7b-chat-hf \
  --target_per_filter_simple 200 \
  --target_per_filter_complex 180 \
  --augment_variants_simple 0 \
  --augment_variants_complex 1 \
  --max_length_simple 256 \
  --max_length_complex 256 \
  --split_ratio 0.9 \
  --reps_per_filter 3 \
  --curriculum_simple_stages "1,2,3" \
  --min_q_chars_simple 12 \
  --min_q_chars_complex 8


Opcionales útiles: --no_strict_simple, --strict_complex, --no_simple_complexity_filter, --allow_other_bases_in_simple.

4) Entrenamiento de ejecutores (train_executors.py)

Propósito: Entrena SIMPLE y COMPLEX con QLoRA 4-bit y evalúa Exact Match por generación desde prefijo (sin teacher forcing).

Entradas:

./data/executors/dataset_simple_{train,test}.json

./data/executors/dataset_complex_{train,test}.json

Salidas:

./models/executors/simple/ y ./models/executors/complex/ (checkpoints)

Métricas y ejemplos:

executor_{name}_metrics.json, executor_{name}_metrics.png

executor_{name}_samples.json, executor_{name}_manual_eval.json

CLI:

python ./train_executors.py \
  --hf_token "$HF_TOKEN" \
  --base_model meta-llama/Llama-2-7b-chat-hf \
  --epochs 10 --batch_size 8 --grad_accum 4 \
  --lr 1e-4 --patience 3 --gen_new_tokens 24

5) Dataset del detector (generate_detector_dataset.py)

Propósito: Construye datasets train/val/OOD para el detector de fugas a partir de (Context, Problem, Answer), generando positivos cuando la Answer contiene (o parafrasea/codifica) el Context.

Entradas:

No requiere ficheros manuales si los textos por filtro se generan en la ruta por defecto. Puedes añadir raíces extra con --txt_roots.

Salidas (por defecto en ./data/detector/):

train.jsonl, val.jsonl, test_ood.jsonl

label_schema.json, stats.json

CLI:

python ./generate_detector_dataset.py \
  --exec_module gen_executors_datasets \
  --exec_module_path ./gen_executors_datasets.py \
  --txt_roots "$DATA_DIR" \
  --target_per_filter 60 \
  --pos_per_neg 3 \
  --val_ratio 0.1 \
  --ood_filters polynomials__add,polynomials__collect,polynomials__compose,polynomials__evaluate,polynomials__expand \
  --seed 42


--dry_run permite ver muestras y estadísticas sin escribir ficheros.

6) Entrenamiento del detector (safety-first) (train_detector.py)

Propósito: Entrena un clasificador (p. ej., RoBERTa-base) y calibra el umbral con política safety_first (prioriza recall objetivo con tope de FPR y banda de incertidumbre).

Entradas:

./data/detector/train.jsonl, val.jsonl, test_ood.jsonl

Salidas (en --output_dir, por defecto ./detector/leakage_classifier/):

Pesos model/ + trainer_log.json

decision_threshold.json (umbral, política, recall objetivo, FPR máximo, margen)

metrics_val.json, metrics_ood.json

confusion_val.json, confusion_ood.json

Curvas: roc_val.png, pr_val.png, roc_ood.png, pr_ood.png

threshold_curve_val.csv, pr_curve_val.csv

Análisis por filtro y tipo: metrics_by_filter_*.json, metrics_by_leaktype_*.json

Ejemplos: examples_val_errors.jsonl, examples_ood_errors.jsonl, smoke_test.json, report.json

CLI:

python ./train_detector.py \
  --model_name roberta-base \
  --output_dir ./models/detector/leakage_classifier \
  --max_length 256 \
  --epochs 3 --batch_size 16 --lr 2e-5 --weight_decay 0.01 --warmup_ratio 0.1 \
  --fp16 \
  --policy safety_first \
  --target_recall 0.995 \
  --max_fpr_cap 0.20 \
  --uncertainty_margin 0.02 \
  --pos_class_weight 2.0

Resultados esperados

Ejecutor SIMPLE: EM = 1.00 en validación.

Ejecutor COMPLEX: EM ≈ 0.89 (épocas 3–5).

Detector (Validación ID): AUROC ≈ 1.00, AUPRC ≈ 1.00, matriz de confusión sin errores al umbral calibrado.

Detector (OOD): Recall ≈ 0.99, AUROC ≈ 0.996, AUPRC ≈ 0.999, con FPR ≈ 0.18 bajo política safety_first.

Los valores pueden variar ligeramente según hardware, semillas y versiones de librerías; el punto de operación está optimizado para recall alto y FPR acotado.

Notas de reproducibilidad

Semilla por defecto: 42 en todos los scripts (puedes cambiarla en CLI).

QLoRA 4-bit: carga en 4-bit con BitsAndBytes; LoRA y padding a la izquierda habilitados en los entrenamientos.

EM por generación: la métrica de los ejecutores se calcula generando a partir de un prefijo de entrada (no teacher forcing), alineando entrenamiento e inferencia.

Umbral del detector: decision_threshold.json documenta la calibración (política, objetivo de recall y margen).
