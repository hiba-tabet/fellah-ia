import sys, json, os
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

classes = json.load(open("export/classes.json"))
sess = ort.InferenceSession("export/model.onnx", providers=["CPUExecutionProvider"])

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

for path in sorted(sys.argv[1:]):
    im = ImageOps.exif_transpose(
        Image.open(path)
    ).convert("RGB").resize((224, 224), Image.BILINEAR)

    x = ((np.asarray(im, dtype=np.float32) / 255 - mean) / std).transpose(2, 0, 1)[None]

    z = sess.run(None, {"image": x})[0][0]

    p = np.exp(z - z.max())
    p /= p.sum()

    top = np.argsort(-p)[:3]

    print(
        os.path.basename(path),
        "|",
        " | ".join(f"{classes[i]} {p[i]:.2f}" for i in top)
    )
