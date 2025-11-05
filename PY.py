from pathlib import Path
from PIL import Image
bad = []
root = Path("data/train3")
exts = {".jpg",".jpeg",".png",".bmp",".webp"}
i = 0
for p in root.rglob("*"):
    i += 1
    if i % 1000 == 0:
        print(i)
    # print(p)
    if p.suffix.lower() in exts:
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception:
            bad.append(p)
print("Broken files:", len(bad))
for b in bad[:50]:
    print(b)