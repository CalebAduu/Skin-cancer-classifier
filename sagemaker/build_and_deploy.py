"""
Package the model + handler into model.tar.gz and deploy a SageMaker Serverless
endpoint. Run from the repo root, with AWS credentials configured.

    python sagemaker/build_and_deploy.py \
        --bundle saved_models/inference_bundle.pt \
        --ood saved_models/ood_reference.pt \
        --role arn:aws:iam::755876201023:role/derm-sagemaker-role \
        --region eu-west-2 --memory-mb 3072

model.tar.gz layout it builds:
    inference_bundle.pt
    ood_reference.pt          (enables the OOD gate; skipped with a warning if absent)
    code/inference.py         (SageMaker handler)
    code/predictor.py         (your app/inference.py, relative imports rewritten)
    code/model.py             (copied from app/)
    code/preprocessing.py     (copied from app/)
    code/scorecam.py          (copied from app/)
    code/requirements.txt
"""
import argparse
import os
import re
import shutil
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.serverless import ServerlessInferenceConfig

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
APP = os.path.join(REPO, "app")


def _rewrite_relative_imports(src: str) -> str:
    # app/inference.py uses `from .model import ...` etc. In the flat code/ dir
    # those must be absolute. Only lines starting with `from .` are touched.
    return re.sub(r"(?m)^from \.", "from ", src)


def build_tarball(bundle_path: str, ood_path: str, out_path: str):
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(bundle_path, os.path.join(tmp, "inference_bundle.pt"))
        if ood_path and os.path.exists(ood_path):
            shutil.copy(ood_path, os.path.join(tmp, "ood_reference.pt"))
            print(f"Including OOD reference: {ood_path}")
        else:
            print("WARNING: no ood_reference.pt found — the OOD gate will be "
                  "inactive on this endpoint (predictions still work, but the "
                  "'not a dermoscopy image' check won't fire). Run "
                  "build_ood_reference.py first to enable it.")

        code = os.path.join(tmp, "code")
        os.makedirs(code)
        shutil.copy(os.path.join(HERE, "code", "inference.py"), code)
        shutil.copy(os.path.join(HERE, "code", "requirements.txt"), code)
        for m in ("model.py", "preprocessing.py", "scorecam.py"):
            shutil.copy(os.path.join(APP, m), code)
        # app/inference.py -> code/predictor.py with absolute imports
        with open(os.path.join(APP, "inference.py")) as f:
            predictor_src = _rewrite_relative_imports(f.read())
        with open(os.path.join(code, "predictor.py"), "w") as f:
            f.write(predictor_src)

        with tarfile.open(out_path, "w:gz") as tar:
            tar.add(tmp, arcname=".")
    print(f"Built {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="saved_models/inference_bundle.pt")
    ap.add_argument("--ood", default="saved_models/ood_reference.pt")
    ap.add_argument("--role", required=True, help="SageMaker execution role ARN")
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "eu-west-2"))
    ap.add_argument("--endpoint-name", default="derm-multimodal-serverless")
    # Verify current tags: https://github.com/aws/deep-learning-containers/blob/master/available_images.md
    ap.add_argument("--framework-version", default="2.1.0")
    ap.add_argument("--py-version", default="py310")
    ap.add_argument("--memory-mb", type=int, default=3072)   # account default quota
    ap.add_argument("--max-concurrency", type=int, default=5)
    args = ap.parse_args()

    boto_sess = boto3.Session(region_name=args.region)
    sess = sagemaker.Session(boto_session=boto_sess)

    tarball = os.path.join(HERE, "model.tar.gz")
    build_tarball(args.bundle, args.ood, tarball)

    model_data = sess.upload_data(path=tarball, key_prefix="derm-multimodal/model")
    print(f"Uploaded to {model_data}")

    model = PyTorchModel(
        model_data=model_data,
        role=args.role,
        entry_point="inference.py",
        framework_version=args.framework_version,
        py_version=args.py_version,
        sagemaker_session=sess,
    )

    serverless_cfg = ServerlessInferenceConfig(
        memory_size_in_mb=args.memory_mb,
        max_concurrency=args.max_concurrency,
    )

    predictor = model.deploy(
        serverless_inference_config=serverless_cfg,
        endpoint_name=args.endpoint_name,
    )
    print(f"\nDeployed serverless endpoint: {predictor.endpoint_name}")
    print("Test with: python sagemaker/invoke_example.py "
          f"--endpoint {predictor.endpoint_name} --image sample.jpg --region {args.region}")


if __name__ == "__main__":
    main()
