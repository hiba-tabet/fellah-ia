import json, time
from pathlib import Path
import numpy as np, torch, timm, onnxruntime as ort
from PIL import Image
from torchvision import transforms as T

# 1) Test de fuite Zeytin (train contre val et test)
sp = json.load(open("data/zeytin_split.json"))
def h(p):
    a = np.asarray(Image.open(p).convert("L").resize((16, 16)), dtype=np.float32).ravel()
    return a > a.mean()
tr = np.stack([h(p) for p, _ in sp["train"]])
for name in ("val", "test"):
    te = np.stack([h(p) for p, _ in sp[name]])
    d = (te[:, None, :] != tr[None, :, :]).sum(2).min(1)
    print(f"FUITE {name}: médiane={np.median(d):.0f} bits | <=10 bits: {(d <= 10).mean():.2f} "
          f"| <=20 bits: {(d <= 20).mean():.2f}")

# 2) Vérification de model.onnx contre best_v4.pth
classes = json.load(open("models/classes.json"))
m = timm.create_model("mobilenetv3_large_100", pretrained=False, num_classes=len(classes))
m.load_state_dict(torch.load("models/best_v4.pth", map_location="cpu")); m.eval()
sess = ort.InferenceSession("export/model.onnx", providers=["CPUExecutionProvider"])
inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
print(f"ONNX entrée: {inp.name} {inp.shape} | sortie: {out.name} {out.shape}")
tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
files = [f for f in sorted(Path("data/prepared/test").rglob("*"))
         if f.suffix.lower() in {".jpg", ".jpeg", ".png"}][::90]
agree, maxdiff, t = 0, 0.0, []
for f in files:
    x = tf(Image.open(f).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        a = torch.softmax(m(x), 1).numpy()
    t0 = time.time()
    b = sess.run(None, {inp.name: x.numpy()})[0]
    t.append(time.time() - t0)
    b = np.exp(b - b.max()); b /= b.sum()
    agree += int(a.argmax() == b.argmax())
    maxdiff = max(maxdiff, float(np.abs(a - b).max()))
print(f"{len(files)} images | même réponse PyTorch/ONNX: {agree}/{len(files)} "
      f"| écart max: {maxdiff:.6f} | temps moyen: {1000 * np.mean(t):.0f} ms")

print("\npreprocessing.json:\n" + open("export/preprocessing.json").read())
