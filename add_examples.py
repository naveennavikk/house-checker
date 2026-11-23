"""Teach the checker from your own examples.

Put photos that should be ACCEPTED in   corrections\\accept\\
and photos that should be REJECTED in   corrections\\reject\\
then run:   .venv\\Scripts\\python add_examples.py

Each photo is added to model/reference_bank.npz (a photo already in the bank is
updated to the new label). Photos that look like your examples then get the same
verdict. Restart the web page (run_web.bat) afterwards so it loads the new examples.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from checker import BANK_PATH, HERE, HouseDoorChecker

FOLDERS = {"accept": 1, "reject": 0}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".jfif", ".webp", ".bmp"}


def main() -> int:
    root = HERE / "corrections"
    photos = [(p, label) for name, label in FOLDERS.items()
              for p in sorted((root / name).glob("*")) if p.suffix.lower() in IMAGE_EXT]
    if not photos:
        for name in FOLDERS:
            (root / name).mkdir(parents=True, exist_ok=True)
        print(f"No photos found. Put photos in {root / 'accept'} or {root / 'reject'} and run this again.")
        return 1

    checker = HouseDoorChecker()
    bank = np.load(BANK_PATH)
    emb, label, names = bank["emb"].astype(np.float32), bank["label"], list(bank["name"])
    for path, y in photos:
        before = checker.check(path)
        e = checker.embed(path)
        dup = np.where(emb @ e > 0.999)[0]  # same photo already in the bank -> relabel it
        if len(dup):
            label[dup] = y
        else:
            emb = np.vstack([emb, e[None].astype(np.float32)])
            label = np.append(label, y)
            names.append(path.name)
        checker.bank_emb, checker.bank_label = emb, label.astype(np.float32)
        after = checker.check(path)
        print(f"{path.name}: {before.label} ({before.probability:.2f}) -> {after.label} ({after.probability:.2f})"
              f"  [taught as {'ACCEPT' if y else 'REJECT'}]")
    np.savez_compressed(BANK_PATH, emb=emb.astype(np.float16), label=label, name=np.array(names))
    print(f"Saved {len(label)} reference photos. Restart the web page to use them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
