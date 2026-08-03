import { NextRequest, NextResponse } from "next/server";
import { SageMakerRuntimeClient, InvokeEndpointCommand } from "@aws-sdk/client-sagemaker-runtime";

export const runtime = "nodejs";

const client = new SageMakerRuntimeClient({
  region: process.env.SM_REGION!,
  credentials: {
    accessKeyId: process.env.SM_ACCESS_KEY_ID!,
    secretAccessKey: process.env.SM_SECRET_ACCESS_KEY!,
  },
});

export async function POST(req: NextRequest) {
  try {
    const form = await req.formData();
    const file = form.get("image") as File | null;
    if (!file) return NextResponse.json({ error: "no image" }, { status: 400 });

    const b64 = Buffer.from(await file.arrayBuffer()).toString("base64");
    const payload = JSON.stringify({
      image_b64: b64,
      age: form.get("age") ? Number(form.get("age")) : null,
      sex: form.get("sex") ?? "unknown",
      localization: form.get("localization") ?? "unknown",
    });

    const res = await client.send(new InvokeEndpointCommand({
      EndpointName: process.env.SM_ENDPOINT!,
      ContentType: "application/json",
      Body: payload,
    }));

    return NextResponse.json(JSON.parse(new TextDecoder().decode(res.Body)));
  } catch (err: any) {
    console.error(err);
    return NextResponse.json({ error: err?.message ?? "inference failed" }, { status: 500 });
  }
}
