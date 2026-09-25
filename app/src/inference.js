// Fellah-IA — moteur de diagnostic hors-ligne

const MODEL_URL = "/model/model.onnx";
const CLASSES_URL = "/model/classes.json";

const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

const CONFIDENCE_THRESHOLD = 0.55;

let session = null;
let classes = null;

async function getSession() {
  if (!session) {
    classes = await (await fetch(CLASSES_URL)).json();

    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"]
    });
  }

  return session;
}

function imageToTensor(img) {
  const canvas = document.createElement("canvas");
  canvas.width = 224;
  canvas.height = 224;

  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, 0, 0, 224, 224);

  const data = ctx.getImageData(0, 0, 224, 224).data;

  const n = 224 * 224;
  const output = new Float32Array(3 * n);

  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) {
      output[k * n + i] =
        (data[i * 4 + k] / 255 - MEAN[k]) / STD[k];
    }
  }

  return new ort.Tensor(
    "float32",
    output,
    [1, 3, 224, 224]
  );
}

function softmax(values) {
  const max = Math.max(...values);

  const exp = values.map(v => Math.exp(v - max));
  const sum = exp.reduce((a, b) => a + b, 0);

  return exp.map(v => v / sum);
}

export async function analyze(input) {
  const modelSession = await getSession();

  let img = input;

  if (!(input instanceof HTMLImageElement)) {
    img = new Image();

    const url = URL.createObjectURL(input);
    img.src = url;

    await img.decode();

    URL.revokeObjectURL(url);
  }

  const tensor = imageToTensor(img);

  const output = await modelSession.run({
    image: tensor
  });

  const probabilities = softmax(
    Array.from(output.logits.data)
  );

  const ranked = probabilities
    .map((confidence, index) => ({
      classId: classes[index],
      confidence
    }))
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 3);

  return {
    confident: ranked[0].confidence >= CONFIDENCE_THRESHOLD,
    results: ranked
  };
}
