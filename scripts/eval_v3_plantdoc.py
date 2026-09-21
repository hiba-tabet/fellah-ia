import json
from pathlib import Path

import torch
import timm
from PIL import Image
from torchvision import transforms as T
from sklearn.metrics import classification_report

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

CLASSES = json.load(open("models/classes.json"))

PD_MAP = {
    "Potato leaf early blight": "potato_early_blight",
    "Potato leaf late blight": "potato_late_blight",
    "Tomato Early blight leaf": "tomato_early_blight",
    "Tomato Septoria leaf spot": "tomato_septoria_leaf_spot",
    "Tomato leaf": "tomato_healthy",
    "Tomato leaf bacterial spot": "tomato_bacterial_spot",
    "Tomato leaf late blight": "tomato_late_blight",
    "Tomato leaf mosaic virus": "tomato_mosaic_virus",
    "Tomato leaf yellow virus": "tomato_yellow_leaf_curl_virus",
    "Tomato mold leaf": "tomato_leaf_mold",
    "Tomato two spotted spider mites leaf": "tomato_spider_mites",
}

EXTS = {".jpg", ".jpeg", ".png", ".webp"}

tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

model = timm.create_model(
    "mobilenetv3_large_100",
    pretrained=False,
    num_classes=len(CLASSES)
)

model.load_state_dict(
    torch.load("models/best_v3.pth", map_location="cpu")
)

model = model.to(DEVICE).eval()

preds = []
labels = []

root = Path("data/plantdoc/test")

for folder, cls in PD_MAP.items():
    d = root / folder
    if not d.exists():
        continue

    true_id = CLASSES.index(cls)

    for f in sorted(d.iterdir()):
        if f.suffix.lower() not in EXTS:
            continue

        img = Image.open(f).convert("RGB")
        x = tf(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = model(x).argmax(1).item()

        labels.append(true_id)
        preds.append(pred)

print("\nRAPPORT PLANTDOC — BEST V3")
print(f"Images évaluées : {len(labels)}")

real_classes = sorted(set(labels))

print(
    classification_report(
        labels,
        preds,
        labels=real_classes,
        target_names=[CLASSES[i] for i in real_classes],
        digits=3,
        zero_division=0
    )
)
