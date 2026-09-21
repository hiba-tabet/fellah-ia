import json, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

random.seed(0)
sp = json.load(open("data/zeytin_split.json"))
EXTS = {".jpg", ".jpeg", ".png"}

def vec(p):
    a = np.asarray(Image.open(p).convert("L").resize((32, 32)), dtype=np.float32)
    outs = []
    for v in (a, a[:, ::-1], a[::-1, :], a[::-1, ::-1]):
        v = v.ravel() - v.mean()
        outs.append(v / (np.linalg.norm(v) + 1e-6))
    return np.stack(outs)

tr_p = [p for p, _ in sp["train"]]
TR = np.stack([vec(p)[0] for p in tr_p])

def best(paths):
    V = np.stack([vec(p) for p in paths])
    S = np.einsum("nvd,md->nvm", V, TR).max(1)
    return S.max(1), S.argmax(1)

def report(name, s):
    print(f"{name:10s} n={len(s):3d} | corrélation max médiane={np.median(s):.2f} | "
          f">=0.90: {(s >= 0.90).mean():.2f} | >=0.95: {(s >= 0.95).mean():.2f}")

ctrl = [f for f in Path("data/prepared/train").glob("tomato_*/*") if f.suffix.lower() in EXTS]
report("TÉMOIN", best(random.sample(ctrl, 200))[0])
tp = [p for p, _ in sp["test"]]
s, idx = best(tp)
report("val", best([p for p, _ in sp["val"]])[0])
report("test", s)

order = np.argsort(-s)
pick = list(order[:6]) + random.sample(list(order[6:]), 6)
T, cols = 160, 3
rows = (len(pick) + cols - 1) // cols
sheet = Image.new("RGB", (cols * 2 * T, rows * (T + 16)), "white")
d = ImageDraw.Draw(sheet)
for k, i in enumerate(pick):
    r, c = divmod(k, cols)
    x0, y0 = c * 2 * T, r * (T + 16)
    sheet.paste(Image.open(tp[i]).convert("RGB").resize((T, T)), (x0, y0))
    sheet.paste(Image.open(tr_p[idx[i]]).convert("RGB").resize((T, T)), (x0 + T, y0))
    d.text((x0 + 4, y0 + T + 2), f"{'TOP' if k < 6 else 'alea'} corr={s[i]:.2f}", fill="black")
sheet.save("data/leak_check.jpg")
print("image enregistrée: data/leak_check.jpg (gauche = test, droite = image la plus proche du train)")
