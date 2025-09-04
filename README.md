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
