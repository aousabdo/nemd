"""Download a minimal CWRU bearing-fault subset.

We pull four classes x 2 files each from the Case Western Reserve
Bearing Data Center:
    normal baseline, inner race fault, ball fault, outer race fault
all at drive-end sampling (12 kHz), 0.007" fault diameter.

This is the de-facto standard 4-class CWRU task that appears in
dozens of EMD/VMD applied papers; we mirror the convention so
NAFB's performance is directly comparable to the published
literature's reported numbers.

Output: data/cwru_4class/{class}/{file}.mat
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

BASE = "https://engineering.case.edu/sites/default/files"

# File IDs follow CWRU's official numbering:
#   https://engineering.case.edu/bearingdatacenter/apparatus-and-procedures
# All are drive-end data at 12 kHz, 0.007" fault, motor loads 0-3 hp.
FILES = {
    "normal":      ["97.mat",  "98.mat",  "99.mat",  "100.mat"],
    "inner_race":  ["105.mat", "106.mat", "107.mat", "108.mat"],
    "ball":        ["118.mat", "119.mat", "120.mat", "121.mat"],
    "outer_race":  ["130.mat", "131.mat", "132.mat", "133.mat"],
}


def main() -> None:
    out = Path("data/cwru_4class")
    out.mkdir(parents=True, exist_ok=True)
    for cls, files in FILES.items():
        cls_dir = out / cls
        cls_dir.mkdir(exist_ok=True)
        for fn in files:
            dest = cls_dir / fn
            if dest.exists() and dest.stat().st_size > 1_000_000:
                print(f"skip  {cls}/{fn}")
                continue
            url = f"{BASE}/{fn}"
            print(f"get   {cls}/{fn} ... ", end="", flush=True)
            try:
                with urllib.request.urlopen(url, timeout=30) as r, \
                     open(dest, "wb") as f:
                    shutil.copyfileobj(r, f)
                print(f"ok ({dest.stat().st_size // 1024} KB)")
            except Exception as e:
                print(f"FAIL ({e})")
                if dest.exists():
                    dest.unlink()


if __name__ == "__main__":
    main()
