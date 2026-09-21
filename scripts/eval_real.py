import sys, json
from pathlib import Path
import torch, timm
from PIL import Image
from torchvision import transforms as T

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
classes = json.load(open("models/classes.json"))

tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def load():
    m = timm.create_model(
        "mobilenetv3_large_100",
        pretrained=False,
        num_classes=len(classes)
    )

    m.load_state_dict(
        torch.load(
            "models/best_v3.pth",
            map_location="cpu"
        )
    )

    return m.to(device).eval()


def predict(m, path):
    x = tf(
        Image.open(path).convert("RGB")
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        p = torch.softmax(m(x), 1)[0]

    top = p.topk(3)

    return [
        (classes[i], float(v))
        for v, i in zip(top.values, top.indices)
    ]


RULES = [
    ("septoria", "tomato_septoria_leaf_spot"),
    ("bacterial", "tomato_bacterial_spot"),
    ("early blight", "tomato_early_blight"),
    ("late blight", "tomato_late_blight"),
    ("mold", "tomato_leaf_mold"),
    ("mosaic", "tomato_mosaic_virus"),
    ("yellow", "tomato_yellow_leaf_curl_virus"),
    ("spider", "tomato_spider_mites")
]


def plantdoc_label(name):
    n = name.lower().strip()

    if "tomato" in n:
        for key, cls in RULES:
            if key in n:
                return cls

        if n == "tomato leaf":
            return "tomato_healthy"

    if "potato" in n:
        if "early" in n:
            return "potato_early_blight"

        if "late" in n:
            return "potato_late_blight"

    return None


def run_plantdoc(m):
    root = Path("data/plantdoc")
    per = {}
    mapped = {}
    skipped = set()

    for split in ("train", "test"):
        for d in sorted((root / split).iterdir()):
            if not d.is_dir():
                continue

            true = plantdoc_label(d.name)

            if true is None:
                skipped.add(d.name)
                continue

            mapped[d.name] = true

            for f in d.iterdir():
                if f.suffix.lower() not in EXTS:
                    continue

                top = predict(m, f)

                r = per.setdefault(true, [0, 0, 0])

                r[0] += 1
                r[1] += top[0][0] == true
                r[2] += true in [c for c, _ in top]

    print("ASSOCIATION dossier PlantDoc -> classe:")

    for k, v in sorted(mapped.items()):
        print(f"  {k}  ->  {v}")

    print("IGNORÉS:", sorted(skipped))

    print(f"\n{'classe':32s}    n  top1  top3")

    for c in sorted(per):
        n, a, b = per[c]

        print(
            f"{c:32s} {n:4d} {a/n:5.2f} {b/n:5.2f}"
        )

    N = sum(v[0] for v in per.values())

    print(
        f"\nTOTAL n={N}  "
        f"top1={sum(v[1] for v in per.values())/N:.3f}"
        f"  top3={sum(v[2] for v in per.values())/N:.3f}"
    )


def run_photos(m):
    root = Path("data/my_photos")
    ok = 0
    n = 0

    for d in sorted(
        p for p in root.iterdir()
        if p.is_dir()
    ):
        labeled = d.name in classes

        for f in sorted(d.iterdir()):
            if f.suffix.lower() not in EXTS:
                continue

            top = predict(m, f)

            mark = ""

            if labeled:
                hit = top[0][0] == d.name
                n += 1
                ok += hit
                mark = "OK" if hit else "ERR"

            print(
                f"{mark:3s} "
                f"{d.name:26s} "
                f"{f.name:20s} -> "
                f"{top[0][0]} ({top[0][1]:.2f}) | "
                f"{top[1][0]} ({top[1][1]:.2f})"
            )

    if n:
        print(
            f"\nPHOTOS étiquetées : "
            f"{ok}/{n} = {ok/n:.3f}"
        )


if __name__ == "__main__":
    model = load()

    if sys.argv[1] == "plantdoc":
        run_plantdoc(model)
    else:
        run_photos(model)
