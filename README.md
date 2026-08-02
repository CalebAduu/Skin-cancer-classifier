---
title: Dermoscopy Multimodal Classifier
emoji: 🩺
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# Multimodal Dermoscopy Classifier — End-to-End Serving

> **Decision-support research prototype, not a diagnostic device.** Trained on
> HAM10000 (+ BCN20000 minority boost); performance degrades under real-world
> domain shift (see *Limitations*). Not for clinical use.

A hybrid CNN–Vision Transformer (`vit_base_r50_s16_224`) that classifies a
dermoscopy image into 7 lesion types by fusing the image with 12 patient-metadata
tokens through one-way cross-attention adapters. This repo takes the model out of
the research notebook it was trained in and turns it into a reproducible,
deployable, calibrated inference service with visual explanations.

## Why this repo exists (the interesting part)
The trained checkpoint alone is **not deployable**: the preprocessing vocabulary,
age-normalisation statistics, and calibration constants (temperature + per-class
Youden thresholds) lived only in notebook memory. The core engineering step here
was defining a **self-contained inference bundle** (`export_bundle.py`) that pairs
the weights with everything needed to reproduce a prediction, then building a
serving layer that reads exclusively from it.

## Architecture
```
image ──► CNN-ViT stem ──► patch tokens ─┐
                                         ├─► one-way cross-attn adapters ─► head ─► logits
age/sex/site ──► 12 metadata tokens ─────┘         (blocks 10–11)             │
                                                                              ▼
                                              temperature scaling + Youden thresholds
                                                                              │
                                                                              ▼
                                              calibrated probabilities + Score-CAM
```
*(replace with a proper diagram)*

## Quickstart
```bash
pip install -r requirements.txt
# 1) create the bundle from the notebook (see export_bundle.py docstring)
# 2) serve
BUNDLE_PATH=saved_models/inference_bundle.pt uvicorn app.main:app --reload
curl -F image=@sample.jpg -F age=60 -F sex=male -F localization=back \
     -F explain=true localhost:8000/predict
```

## Results
| Metric | Test (HAM10000) | External (Derm7pt) | External (PAD-UFES-20) |
|---|---|---|---|
| Macro AUROC | _fill_ | _fill_ | _fill_ |
| Malignant sensitivity (argmax) | _fill_ | _fill_ | _fill_ |
| Malignant sensitivity (Youden) | _fill_ | _fill_ | _fill_ |
| ECE (after temp. scaling) | _fill_ | — | — |

_Pull these from the notebook — do not inflate. Report both argmax and
Youden-recalibrated numbers._

## Serving metrics
| | value |
|---|---|
| p50 / p95 latency (CPU) | _fill_ |
| p50 / p95 latency (GPU) | _fill_ |
| Cost per 1k inferences | _fill_ |
| Model size on disk | _fill_ |

## Limitations (read first)
- Trained on dermoscopic images; **calibration does not transfer** under domain
  shift — external cohorts hold ranking (AUROC) but need re-thresholding.
- Clinical (smartphone) images are a severe shift; treat outputs as low-confidence.
- Two lesion-level metadata features default at single-image inference time.
- Not evaluated for, or intended for, clinical deployment.

## Full thesis
_link to the thesis PDF_
