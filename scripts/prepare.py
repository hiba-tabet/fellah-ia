import json, os, random, shutil
from pathlib import Path

random.seed(42)
ROOT = Path.cwd()
PV = ROOT / "data/plantvillage/plantvillage dataset/color"
OL = ROOT / "data/olive/dataset"
OUT = ROOT / "data/prepared"
EXTS = {".jpg", ".jpeg", ".png"}
CAP_TRAIN = 700

PV_CLASSES = {
    "Tomato___Bacterial_spot": "tomato_bacterial_spot",
    "Tomato___Early_blight": "tomato_early_blight",
    "Tomato___Late_blight": "tomato_late_blight",
    "Tomato___Leaf_Mold": "tomato_leaf_mold",
    "Tomato___Septoria_leaf_spot": "tomato_septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "tomato_spider_mites",
    "Tomato___Target_Spot": "tomato_target_spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "tomato_yellow_leaf_curl_virus",
    "Tomato___Tomato_mosaic_virus": "tomato_mosaic_virus",
    "Tomato___healthy": "tomato_healthy",
    "Potato___Early_blight": "potato_early_blight",
    "Potato___Late_blight": "potato_late_blight",
    "Potato___healthy": "potato_healthy",
}

OL_CLASSES = {
    "Healthy": "olive_healthy",
    "aculus_olearius": "olive_aculus_olearius",
    "olive_peacock_spot": "olive_peacock_spot",
}

def imgs(d):
    return sorted(f for f in Path(d).iterdir() if f.suffix.lower() in EXTS)

def link(files, split, cls):
    d = OUT / split / cls
    d.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(files):
        os.symlink(f.resolve(), d / f"{i:05d}_{f.name}")

if OUT.exists():
    shutil.rmtree(OUT)

stats = {}

for src, cls in PV_CLASSES.items():
    files = imgs(PV / src)
    random.shuffle(files)

    n_test = int(0.15 * len(files))
    n_val = int(0.15 * len(files))

    test = files[:n_test]
    val = files[n_test:n_test + n_val]
    train = files[n_test + n_val:][:CAP_TRAIN]

    for s, fl in (("train", train), ("val", val), ("test", test)):
        link(fl, s, cls)

    stats[cls] = (len(train), len(val), len(test))

for src, cls in OL_CLASSES.items():
    tr = imgs(OL / "train" / src)
    random.shuffle(tr)

    n_val = int(0.15 * len(tr))
    val = tr[:n_val]
    train = tr[n_val:][:CAP_TRAIN]
    test = imgs(OL / "test" / src)

    for s, fl in (("train", train), ("val", val), ("test", test)):
        link(fl, s, cls)

    stats[cls] = (len(train), len(val), len(test))

classes = sorted(stats)

json.dump(classes, open(OUT / "classes.json", "w"), indent=2)

print(f"{'classe':34s} train   val  test")
for c in classes:
    t, v, te = stats[c]
    print(f"{c:34s} {t:5d} {v:5d} {te:5d}")

print("CLASSES:", len(classes))
print("TOTAL train/val/test:",
      *[sum(stats[c][i] for c in classes) for i in range(3)])
