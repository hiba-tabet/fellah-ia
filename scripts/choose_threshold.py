import json
from pathlib import Path
import torch
import timm
from PIL import Image
from torchvision import transforms as T
import sys

sys.path.insert(0, "scripts")
from eval_real import plantdoc_label

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

classes = json.load(open("export/classes.json"))

tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

EXTS = {".jpg", ".jpeg", ".png", ".webp"}

m = timm.create_model(
    "mobilenetv3_large_100",
    pretrained=False,
    num_classes=len(classes)
)

m.load_state_dict(
    torch.load("models/best_v4.pth", map_location="cpu")
)

m = m.to(device).eval()

def collect():
    items = []
    for d in sorted(Path("data/plantdoc/test").iterdir()):
        if d.is_dir():
            c = plantdoc_label(d.name)
            if c and c in classes:
                items += [
                    (f, classes.index(c))
                    for f in d.iterdir()
                    if f.suffix.lower() in EXTS
                ]
    return items

items = collect()

print(f"n={len(items)} (PlantDoc test)")

if not items:
    raise RuntimeError("Aucune image PlantDoc trouvée dans data/plantdoc/test")

probs, ys = [], []

with torch.no_grad():
    for f, y in items:
        x = tf(Image.open(f).convert("RGB")).unsqueeze(0).to(device)
        probs.append(torch.softmax(m(x), 1)[0].cpu())
        ys.append(y)

probs = torch.stack(probs)
ys = torch.tensor(ys)

conf = probs.max(1).values
pred = probs.argmax(1)

print()
print(f"{'seuil':>6s} {'couverture':>11s} {'précision':>15s} {'refus':>12s}")

for t in (0.0, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9):
    keep = conf >= t
    cov = keep.float().mean().item()

    if keep.any():
        acc = (pred[keep] == ys[keep]).float().mean().item()
    else:
        acc = float("nan")

    print(f"{t:6.1f} {cov:11.2%} {acc:15.2%} {1-cov:12.2%}")
