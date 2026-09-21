import sys, random, shutil, json
from pathlib import Path
sys.path.insert(0, "scripts")
from eval_real import plantdoc_label

random.seed(1)
EXTS = {".jpg", ".jpeg", ".png"}
out = Path("export/samples")
shutil.rmtree(out, ignore_errors=True)
out.mkdir(parents=True)

pd = [f for d in Path("data/plantdoc/test").iterdir()
      if d.is_dir() and plantdoc_label(d.name)
      for f in d.iterdir() if f.suffix.lower() in EXTS]

z = [Path(p) for p, _ in json.load(open("data/zeytin_split.json"))["test"]]

pv = [f for f in Path("data/prepared/test").rglob("*")
      if f.suffix.lower() in EXTS]

for tag, pool in (("plantdoc", pd), ("zeytin", z), ("labo", pv)):
    for i, p in enumerate(random.sample(pool, 4)):
        shutil.copy(p.resolve(), out / f"{tag}_{i}{p.suffix.lower()}")

print(sorted(x.name for x in out.iterdir()))
