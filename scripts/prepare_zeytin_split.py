import json, random
from pathlib import Path
from PIL import Image

random.seed(42)
EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def ahash(p):
    im = Image.open(p).convert("L").resize((8, 8))
    px = list(im.getdata())
    avg = sum(px) / 64
    return "".join("1" if v > avg else "0" for v in px)

groups = {}

for label in ("healthy", "diseased"):
    for f in sorted((Path("data/olive_binary") / label).rglob("*")):
        if f.suffix.lower() in EXTS:
            groups.setdefault(ahash(f), []).append(
                (str(f.resolve()), label)
            )

gl = list(groups.values())
random.shuffle(gl)

total = sum(len(g) for g in gl)

split = {
    "train": [],
    "val": [],
    "test": []
}

acc = 0

for g in gl:
    r = acc / total

    if r < 0.5:
        s = "train"
    elif r < 0.7:
        s = "val"
    else:
        s = "test"

    split[s] += g
    acc += len(g)

json.dump(
    split,
    open("data/zeytin_split.json", "w"),
    ensure_ascii=False
)

print(
    "images:", total,
    "| groupes:", len(gl),
    "| groupes de quasi-doublons:",
    sum(len(g) > 1 for g in gl),
    "| plus gros groupe:",
    max(len(g) for g in gl)
)

for s, items in split.items():
    print(
        f"{s:5s} n={len(items):4d}"
        f"  saines={sum(l == 'healthy' for _, l in items):4d}"
        f"  malades={sum(l == 'diseased' for _, l in items):4d}"
    )
