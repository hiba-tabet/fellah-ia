import * as ort from "onnxruntime-web/wasm";


const MODEL_URL = "/model/model.onnx";
const CLASSES_URL = "/model/classes.json";

const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const CONFIDENCE_THRESHOLD = 0.55;

let session: ort.InferenceSession | null = null;
let classes: string[] | null = null;

async function getSession(): Promise<ort.InferenceSession> {
  if (!session) {
    const response = await fetch(CLASSES_URL);
    classes = (await response.json()) as string[];

    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"],
    });
  }

  return session;
}

function imageToTensor(img: HTMLImageElement): ort.Tensor {
  const canvas = document.createElement("canvas");
  canvas.width = 224;
  canvas.height = 224;

  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Impossible de créer le contexte canvas.");
  }

  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, 0, 0, 224, 224);

  const data = ctx.getImageData(0, 0, 224, 224).data;
  const n = 224 * 224;
  const output = new Float32Array(3 * n);

  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) {
      const pixel = data[i * 4 + k] ?? 0;
      output[k * n + i] =
        (pixel / 255 - (MEAN[k] ?? 0)) / (STD[k] ?? 1);
    }
  }

  return new ort.Tensor("float32", output, [1, 3, 224, 224]);
}

function softmax(values: number[]): number[] {
  const max = Math.max(...values);
  const exp = values.map((v) => Math.exp(v - max));
  const sum = exp.reduce((a, b) => a + b, 0);

  return exp.map((v) => v / sum);
}

export async function analyze(
  input: string | Blob | HTMLImageElement,
) {
  const modelSession = await getSession();

  let img: HTMLImageElement;

  if (input instanceof HTMLImageElement) {
    img = input;
  } else {
    img = new Image();

    if (typeof input === "string") {
      img.src = input;
      await img.decode();
    } else {
      const url = URL.createObjectURL(input);
      img.src = url;

      try {
        await img.decode();
      } finally {
        URL.revokeObjectURL(url);
      }
    }
  }

  const tensor = imageToTensor(img);

  const output = await modelSession.run({
    image: tensor,
  });

  const logits = output["logits"];

  if (!logits || !(logits.data instanceof Float32Array)) {
    throw new Error("Sortie du modèle ONNX invalide.");
  }

  const probabilities = softmax(Array.from(logits.data));

  if (!classes) {
    throw new Error("Classes du modèle introuvables.");
  }

  const ranked = probabilities
    .map((confidence, index) => ({
      classId: (classes ?? [])[index] ?? "",
      confidence,
    }))
    .filter((result) => result.classId !== "")
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 3);

  console.log("FELLAH AI RESULTS:", ranked);

  return {
    confident: (ranked[0]?.confidence ?? 0) >= CONFIDENCE_THRESHOLD,
    results: ranked,
  };
}
