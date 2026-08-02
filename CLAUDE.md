# CLAUDE.md — Dermoscopy Multimodal Classifier (serving repo)

## What this is
Serving + inference wrapper around an MSc-thesis model: a hybrid CNN-ViT
(`vit_base_r50_s16_224`) that fuses a dermoscopy image with 12 patient-metadata
tokens via one-way cross-attention adapters. This repo takes the trained model
out of a research notebook and makes it a reproducible, deployable service.

**Framing:** decision-support research prototype, NOT a diagnostic device. Never
add copy that implies clinical readiness. Lead with limitations.

## Champion config (do not change without re-exporting the bundle)
- `adapter_type='oneway'`, `adapter_layers=2`, `use_derm_token=False`
- 7 classes, order: NV, MEL, BCC, AKIEC, BKL, DF, VASC
- Derm Foundation token is OFF — no Derm embeddings needed at inference.

## The key artifact: the inference bundle
The raw `.pth` is NOT sufficient to serve. `saved_models/inference_bundle.pt`
must also carry: `cat_maps`, `cat_cardinalities`, `meta_num_cols/meta_cat_cols`,
`age_mean/age_std/age_median`, `temperature` (T_opt), `per_class_thresholds`,
`class_names`. Produce it by running `export_bundle.py::export()` inside the
thesis notebook after the temperature + Youden cells. Everything downstream
reads from this bundle — never recompute preprocessing state elsewhere.

## Layout
- `app/model.py` — model classes (extracted verbatim from the notebook).
- `app/preprocessing.py` — single-sample feature engineering + transform + encode.
- `app/scorecam.py` — Score-CAM (targets ResNet stage-2 last block).
- `app/inference.py` — `Predictor`: bundle+weights -> calibrated prediction.
- `app/main.py` — FastAPI (`/predict`, `/health`).
- `export_bundle.py` — run in the notebook to create the bundle.

## Calibration (reproduce notebook exactly)
1. `softmax(logits / T)` with the fitted temperature.
2. Youden per-class thresholds with prob/threshold-ratio tie-break -> a
   `calibrated_label` alongside plain `argmax_label`.

## Inference-time modelling note (keep honest)
`images_per_lesion` and `is_multi_image_lesion` are lesion-level (groupby over the
training set) and cannot be computed for a novel single image, so they default to
1 and 0. Document this in the model card.

## Commands
- Serve: `uvicorn app.main:app --reload`
- Health: `curl localhost:8000/health`
- Predict: `curl -F image=@sample.jpg -F age=60 -F sex=male -F localization=back localhost:8000/predict`

## Definition of done for the current milestone
Bundle exported -> `Predictor` loads it and reproduces the notebook's test-set
predictions on a few sample images (sanity: same argmax as the notebook) ->
`/predict` returns calibrated probs + latency -> Dockerfile -> one deployed
endpoint with p50/p95 latency and cost/1k inferences in the README.
