import sys, json
from pathlib import Path
from collections import Counter
import torch
import timm
from PIL import Image
from torchvision import transforms as T

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
classes = json.load(open("models/classes.json"))

olive = {i for i, c in enumerate(classes) if c.startswith("olive_")}
H = classes.index("olive_healthy")
D = [
    classes.index("olive_aculus_olearius"),
    classes.index("olive_peacock_spot")
]

tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

EXTS = {".jpg", ".jpeg", ".png", ".webp"}

path = sys.argv[1] if len(sys.argv) > 1 else "models/best_v3.pth"

m = timm.create_model(
    "mobilenetv3_large_100",
    pretrained=False,
    num_classes=len(classes)
)
m.load_state_dict(torch.load(path, map_location="cpu"))
m = m.to(device).eval()

recalls = {}

for label in ("healthy", "diseased"):
    files = [
        f for f in (Path("data/olive_binary") / label).rglob("*")
        if f.suffix.lower() in EXTS
    ]

    crop_ok = 0
    bin_ok = 0
    dist = Counter()

    for f in files:
        x = tf(Image.open(f).convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            p = torch.softmax(m(x), 1)[0].cpu()

        top = int(p.argmax())
        dist[classes[top]] += 1

        crop_ok += top in olive

        p_h = float(p[H])
        p_d = float(sum(p[i] for i in D))

        if label == "healthy":
            bin_ok += p_h > p_d
        else:
            bin_ok += p_d > p_h

    n = len(files)
    recalls[label] = bin_ok / n

    print(f"\n[{label}] n={n}")
    print(f"  culture reconnue (top1 = une classe d'olivier): {crop_ok/n:.3f}")
    print(f"  saine/malade correct (p_saine vs p_malade):     {bin_ok/n:.3f}")

    for c, k in dist.most_common(6):
        print(f"    top1 = {c:32s} {k:4d} ({k/n:.2f})")

print(
    f"\nPrécision équilibrée saine/malade = "
    f"{(recalls['healthy'] + recalls['diseased'])/2:.3f}"
)
