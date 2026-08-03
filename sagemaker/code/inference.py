"""
SageMaker inference handler.

This reuses YOUR own Predictor (app/inference.py, packaged here as predictor.py)
so the endpoint returns exactly what the FastAPI backend and the web UI expect —
including the OOD fields (ood_distance, ood_threshold, likely_out_of_distribution).

Score-CAM is intentionally disabled here: it runs many forward passes and would
risk the CPU serverless timeout, so `explain` is forced False regardless of the
request. The heatmap panel in the UI simply won't populate via this endpoint.
"""
import io
import json
import base64

from PIL import Image

# predictor.py is app/inference.py with its relative imports rewritten to
# absolute — the build script handles that. It loads inference_bundle.pt and,
# from the same directory, ood_reference.pt (so the OOD gate is active).
from predictor import Predictor


def model_fn(model_dir):
    return Predictor(f"{model_dir}/inference_bundle.pt", device="cpu")


def input_fn(request_body, content_type):
    if content_type == "application/json":
        data = json.loads(request_body)
        img = Image.open(io.BytesIO(base64.b64decode(data["image_b64"])))
        return {
            "image": img,
            "age": data.get("age"),
            "sex": data.get("sex", "unknown"),
            "localization": data.get("localization", "unknown"),
        }
    if content_type in ("application/x-image", "image/jpeg", "image/png"):
        return {"image": Image.open(io.BytesIO(request_body)),
                "age": None, "sex": "unknown", "localization": "unknown"}
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(inputs, predictor):
    out = predictor.predict(
        inputs["image"],
        age=inputs["age"],
        sex=inputs["sex"],
        localization=inputs["localization"],
        explain=False,  # Score-CAM off on serverless (timeout risk)
    )
    out["disclaimer"] = "Research prototype for decision support only; not a diagnostic device."
    return out


def output_fn(prediction, accept):
    return json.dumps(prediction), "application/json"
