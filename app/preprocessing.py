"""
Inference-time preprocessing.

Mirrors `engineer_ham10000_features` + `val_tfms` + the Dataset encoding from the
notebook, but for a SINGLE raw sample (one image + age/sex/localization) rather
than a dataframe. All state that varies with training (age stats, category
vocabularies) is read from the exported inference bundle — never recomputed here.

Important modelling note carried over honestly:
Two of the 12 features are lesion-level and computed via a groupby over the whole
training set (`images_per_lesion`, `is_multi_image_lesion`). For a novel single
image there is no lesion group, so they default to 1 and 0 respectively. This is
a deliberate inference-time decision, not an oversight — document it in the model
card.
"""
from __future__ import annotations

from typing import Optional

import torch
import torchvision.transforms as T
from PIL import Image

# Same transform used for val/test in the notebook.
INFER_TFMS = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Feature schema — must match the training notebook exactly and in order.
META_COLS_NUM = ["age_approx", "age_normalized", "site_risk_score", "images_per_lesion"]
META_COLS_CAT = [
    "sex", "anatom_site_general", "age_risk_group", "is_high_risk_age",
    "is_sun_exposed", "male_senior", "female_young", "is_multi_image_lesion",
]

_SUN_EXPOSED = {
    "face", "scalp", "ear", "neck", "upper extremity", "hand",
    "lower extremity", "foot", "back", "chest",
}
_SITE_RISK_MAP = {
    "back": 3, "trunk": 3,
    "lower extremity": 2, "upper extremity": 2, "chest": 2, "abdomen": 2,
    "face": 1, "scalp": 1, "neck": 1, "hand": 1, "foot": 1, "acral": 1, "ear": 1,
    "genital": 0, "unknown": 1,
}


def _age_risk_group(age: float) -> str:
    # pd.cut bins=[0,30,50,70,100], labels young/middle/senior/elderly; else unknown
    if age is None:
        return "unknown"
    if 0 < age <= 30:
        return "young"
    if 30 < age <= 50:
        return "middle"
    if 50 < age <= 70:
        return "senior"
    if 70 < age <= 100:
        return "elderly"
    return "unknown"


def build_features(age: Optional[float], sex: str, localization: str, bundle: dict) -> dict:
    """Reproduce the 12 engineered features for one raw sample."""
    age_mean = bundle["age_mean"]
    age_std = bundle["age_std"]
    age_median = bundle.get("age_median", age_mean)

    age_approx = float(age) if age is not None else float(age_median)

    sex = (sex or "unknown").strip().lower()
    if sex in ("", "nan", "none", "na"):
        sex = "unknown"

    site = (localization or "unknown").strip().lower()
    if site in ("", "nan", "none", "na"):
        site = "unknown"

    age_normalized = (age_approx - age_mean) / age_std if age_std else 0.0
    is_high_risk_age = int(age_approx >= 50)
    male_senior = int(sex == "male" and age_approx >= 50)
    female_young = int(sex == "female" and age_approx < 50)
    is_sun_exposed = int(site in _SUN_EXPOSED)
    site_risk_score = int(_SITE_RISK_MAP.get(site, 1))

    return {
        "age_approx": age_approx,
        "age_normalized": age_normalized,
        "site_risk_score": site_risk_score,
        "images_per_lesion": 1.0,          # single-image default (see module docstring)
        "sex": sex,
        "anatom_site_general": site,
        "age_risk_group": _age_risk_group(age_approx),
        "is_high_risk_age": str(is_high_risk_age),
        "is_sun_exposed": str(is_sun_exposed),
        "male_senior": str(male_senior),
        "female_young": str(female_young),
        "is_multi_image_lesion": "0",       # single-image default
    }


def encode_sample(image: Image.Image, age, sex, localization, bundle: dict):
    """Return (x_img[1,3,224,224], x_num[1,4], x_cat[1,8]) ready for the model."""
    feats = build_features(age, sex, localization, bundle)
    cat_maps = bundle["cat_maps"]

    x_img = INFER_TFMS(image.convert("RGB")).unsqueeze(0)

    num = [float(feats[c]) for c in META_COLS_NUM]
    x_num = torch.tensor(num, dtype=torch.float32).unsqueeze(0)

    cat = []
    for c in META_COLS_CAT:
        val = str(feats[c])
        if val not in cat_maps[c]:
            val = "unknown"
        cat.append(cat_maps[c][val])
    x_cat = torch.tensor(cat, dtype=torch.long).unsqueeze(0)

    return x_img, x_num, x_cat
