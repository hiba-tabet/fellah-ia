from pathlib import Path
import shutil

SRC = Path("data/zeytin")
OUT = Path("data/olive_binary")

mapping = {
    "sağlam": "healthy",
    "hastalıklı": "diseased",
}

for src_name, out_name in mapping.items():
    src = SRC / src_name
    dst = OUT / out_name
    dst.mkdir(parents=True, exist_ok=True)

    files = [p for p in src.rglob("*") if p.is_file()]
    print(f"{src_name} -> {out_name}: {len(files)} images")

    for i, f in enumerate(files):
        target = dst / f"{i:05d}_{f.name}"
        if not target.exists():
            shutil.copy2(f, target)

print("\nTerminé.")
