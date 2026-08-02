"""
Inference entry point. Loads the model + the exported inference bundle, and
exposes a single `Predictor.predict(...)` returning calibrated probabilities.

Calibration reproduces the notebook exactly:
  1. raw logits -> temperature scaling with T_opt  (softmax(logits / T))
  2. Youden per-class thresholds with the prob/threshold ratio tie-break, giving
     a `calibrated_label` alongside the plain `argmax_label`.
Temperature scaling does not change argmax; the Youden step can.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .model import ViT_HAM10000_HybridCNN, CHAMPION_CONFIG, CLASS_NAMES, MALIGNANT_IDX
from .preprocessing import encode_sample
from .scorecam import ScoreCAM


def _scaled_probs(logits: torch.Tensor, T: float) -> np.ndarray:
    return F.softmax(logits / T, dim=1).cpu().numpy()


class Predictor:
    def __init__(self, bundle_path: str, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        self.bundle = bundle

        cfg = dict(CHAMPION_CONFIG)
        cfg["num_num"] = len(bundle["meta_num_cols"])
        cfg["cat_cardinalities"] = bundle["cat_cardinalities"]
        cfg["pretrained"] = False

        self.model = ViT_HAM10000_HybridCNN(**cfg)
        self.model.load_state_dict(bundle["model_state_dict"])
        self.model.to(self.device).eval()

        self.T = float(bundle.get("temperature", 1.0))
        thr = bundle.get("per_class_thresholds")
        self.thresholds = np.asarray(thr, dtype=float) if thr is not None else None
        self.class_names = bundle.get("class_names", CLASS_NAMES)

        # Out-of-distribution gate: the model is a closed-set classifier, so
        # softmax always returns a confident-looking distribution over the 7
        # trained classes even for a completely unrelated image (a photo of a
        # cat, a document, ...). Rather than trust the model's own confidence
        # (deep nets are frequently overconfident on garbage inputs), compare
        # the image's own embedding against a small reference bank of known
        # real dermoscopy images — see build_ood_reference.py. Optional: if
        # the reference file isn't present, the gate is silently skipped.
        self.ood_bank = None
        self.ood_threshold = None
        ood_path = Path(bundle_path).parent / "ood_reference.pt"
        if ood_path.exists():
            ood = torch.load(ood_path, map_location="cpu", weights_only=False)
            self.ood_bank = ood["embeddings"].to(self.device)
            self.ood_threshold = float(ood["threshold"])

    @torch.no_grad()
    def predict(self, image: Image.Image, age=None, sex="unknown",
                localization="unknown", explain: bool = False) -> dict:
        x_img, x_num, x_cat = encode_sample(image, age, sex, localization, self.bundle)
        x_img = x_img.to(self.device)
        x_num = x_num.to(self.device)
        x_cat = x_cat.to(self.device)

        logits = self.model(x_img, x_num, x_cat, None)
        probs = _scaled_probs(logits, self.T)[0]

        argmax_idx = int(probs.argmax())
        result = {
            "argmax_label": self.class_names[argmax_idx],
            "argmax_index": argmax_idx,
            "probabilities": {self.class_names[i]: float(probs[i]) for i in range(len(probs))},
            "malignant_probability": float(sum(probs[i] for i in MALIGNANT_IDX)),
            "temperature": self.T,
        }

        if self.thresholds is not None:
            ratios = probs / self.thresholds
            above = np.where(probs >= self.thresholds)[0]
            cal_idx = int(above[ratios[above].argmax()]) if len(above) else int(ratios.argmax())
            result["calibrated_label"] = self.class_names[cal_idx]
            result["calibrated_index"] = cal_idx

        if self.ood_bank is not None:
            img_emb = F.normalize(self.model.embed_image(x_img), dim=1)
            sims = (img_emb @ self.ood_bank.T)[0]
            nn_dist = float(1.0 - sims.max())
            result["ood_distance"] = nn_dist
            result["ood_threshold"] = self.ood_threshold
            result["likely_out_of_distribution"] = nn_dist > self.ood_threshold

        if explain:
            result["scorecam_png_base64"] = self._scorecam_overlay(image, x_img, x_num, x_cat, argmax_idx)
        return result

    def _scorecam_overlay(self, orig_image, x_img, x_num, x_cat, target_idx) -> str:
        cam_engine = ScoreCAM(self.model)
        try:
            cam, _, _ = cam_engine.generate_cam(x_img, x_num, x_cat, None, target_class=target_idx)
        finally:
            cam_engine.remove()

        # Overlay heatmap on the 224x224 center-cropped view of the input.
        base = orig_image.convert("RGB").resize((256, 256))
        base = base.crop((16, 16, 240, 240))  # center 224 crop to match INFER_TFMS
        base_arr = np.asarray(base).astype(np.float32) / 255.0

        heat = _colorize(cam)
        overlay = np.clip(0.5 * base_arr + 0.5 * heat, 0, 1)
        buf = io.BytesIO()
        Image.fromarray((overlay * 255).astype(np.uint8)).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")


def _colorize(cam: np.ndarray) -> np.ndarray:
    """Minimal jet-like colormap without a hard cv2 dependency."""
    try:
        import cv2
        heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    except Exception:
        import matplotlib.cm as cm
        return cm.get_cmap("jet")(cam)[..., :3].astype(np.float32)
