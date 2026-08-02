"""
Build a small out-of-distribution (OOD) reference bank from known-real
dermoscopy images, so the service can flag inputs that don't look like a
dermoscopy image at all (a photo of a cat, a document, etc.) instead of
confidently forcing them into one of the 7 trained classes.

Run once (or whenever the reference set changes):
    python build_ood_reference.py

Reads verification_set.json (12 real HAM10000 images), embeds each with the
model's image-only CLS token (see ViT_HAM10000_HybridCNN.embed_image), and
calibrates a distance threshold from the *internal* spread of those 12
embeddings (leave-one-out nearest-neighbor cosine distance) with a safety
margin. This is a coarse first-pass calibration — 12 samples is thin for a
statistically tight threshold — documented honestly rather than presented as
more rigorous than it is.
"""
import base64
import io
import json

import torch
import torch.nn.functional as F
from PIL import Image

from app.inference import Predictor
from app.preprocessing import INFER_TFMS

SAFETY_MARGIN = 1.5


def main():
    with open("verification_set.json", "r", encoding="utf-8") as f:
        samples = json.load(f)

    predictor = Predictor("saved_models/inference_bundle.pt")
    model = predictor.model

    embeddings = []
    for s in samples:
        img = Image.open(io.BytesIO(base64.b64decode(s["image_b64"]))).convert("RGB")
        x_img = INFER_TFMS(img).unsqueeze(0)
        emb = model.embed_image(x_img)
        embeddings.append(F.normalize(emb, dim=1)[0])

    bank = torch.stack(embeddings)  # [N, 768], L2-normalized

    # Leave-one-out nearest-neighbor cosine distance within the reference set
    # itself, as a rough estimate of "how far apart do real dermoscopy images
    # get from each other". A new image farther than SAFETY_MARGIN times the
    # worst of these is treated as suspect.
    sims = bank @ bank.T
    sims.fill_diagonal_(-1.0)
    nn_sim = sims.max(dim=1).values
    nn_dist = 1.0 - nn_sim
    internal_spread = float(nn_dist.max())
    threshold = internal_spread * SAFETY_MARGIN

    torch.save(
        {
            "embeddings": bank,
            "threshold": threshold,
            "internal_spread": internal_spread,
            "safety_margin": SAFETY_MARGIN,
            "n_reference_samples": len(samples),
        },
        "saved_models/ood_reference.pt",
    )

    print(f"Reference bank: {len(samples)} embeddings, dim={bank.shape[1]}")
    print(f"Internal leave-one-out spread (max nn distance): {internal_spread:.4f}")
    print(f"OOD threshold ({SAFETY_MARGIN}x margin): {threshold:.4f}")
    print("Wrote saved_models/ood_reference.pt")


if __name__ == "__main__":
    main()
