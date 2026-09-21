import sys
from pathlib import Path

root = Path(sys.argv[1])
exts = {".jpg", ".jpeg", ".png"}
total = 0
for d in sorted(p for p in root.rglob("*") if p.is_dir()):
    n = sum(1 for f in d.iterdir() if f.suffix.lower() in exts)
    if n:
        print(f"{n:6d}  {d.relative_to(root)}")
        total += n
print("TOTAL:", total)
