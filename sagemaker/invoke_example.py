"""
Invoke the deployed SageMaker endpoint with an image + optional metadata.

    python sagemaker/invoke_example.py --endpoint derm-multimodal-serverless \
        --image sample.jpg --age 60 --sex male --localization back --region eu-west-2

To test the OOD gate, pass a non-dermoscopy image (a photo of anything) and look
for "likely_out_of_distribution": true in the response.
"""
import argparse
import base64
import json

import boto3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--age", type=float, default=None)
    ap.add_argument("--sex", default="unknown")
    ap.add_argument("--localization", default="unknown")
    ap.add_argument("--region", default="eu-west-2")
    args = ap.parse_args()

    with open(args.image, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("ascii")

    payload = json.dumps({
        "image_b64": img_b64,
        "age": args.age,
        "sex": args.sex,
        "localization": args.localization,
    })

    rt = boto3.client("sagemaker-runtime", region_name=args.region)
    resp = rt.invoke_endpoint(
        EndpointName=args.endpoint,
        ContentType="application/json",
        Body=payload,
    )
    print(json.dumps(json.loads(resp["Body"].read()), indent=2))


if __name__ == "__main__":
    main()
