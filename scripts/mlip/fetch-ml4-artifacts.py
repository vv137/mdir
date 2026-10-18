#!/usr/bin/env python3
"""Downloads the pretrained checkpoints of ML4's baselines (issue #377,
docs/ml4-baselines.md) and checks each against its pinned SHA-256. Nothing
here is part of MDIR: the files stay where they are written, under their
own licenses, and are never committed.

Each checkpoint is pinned by an immutable URL (a release asset, a commit of a
repository, or an object of a store) and by its hash; a file whose hash
differs is removed and the script fails. A file already present with the
right hash is not downloaded again.

usage: fetch-ml4-artifacts.py DIR [NAME ...] [--list]

NAME is one of the names that `--list` prints; without names, the
recommended pair (painn-nabladft-100k and mace-off23-medium) is fetched.
The metatomic artifacts are then made with `painn-to-metatomic.py` and
`mace-to-metatomic.py`.
"""
import argparse
import hashlib
import os
import sys
import urllib.request

ARTIFACTS = {
    # SchNetPack 2 PaiNN of the nablaDFT benchmark, training split
    # "train_large" (nablaDFT's models_checkpoints.json, key
    # PaiNN_train_large); the store's ETag (MD5) is
    # 8aff6cdaa4d4698441584ba86f6ed27e. Code MIT (AIRI-Institute/nablaDFT).
    "painn-nabladft-100k": dict(
        url="https://a002dlils-kadurin-nabladft.obs.ru-moscow-1.hc.sbercloud.ru/"
            "data/nablaDFTv2/models_checkpoints/PaiNN/painn_100k.ckpt",
        file="painn_100k.ckpt",
        size=5398754,
        sha256="6c81c40a99b2618ae4c0e5f96d441ece5c12bf47e1d8137cbd62883bcf77ef06",
        license="MIT (nablaDFT repository)",
    ),
    # MACE-OFF23 medium, ACEsuit/mace-off at commit 91a78c5 (2025-01-29).
    # Academic Software License: academic, non-commercial use.
    "mace-off23-medium": dict(
        url="https://raw.githubusercontent.com/ACEsuit/mace-off/"
            "91a78c5a9c300d1104700d9352c8bfe449227737/mace_off23/MACE-OFF23_medium.model",
        file="MACE-OFF23_medium.model",
        size=18350596,
        sha256="4842c52ad210d6e1f84d6cf1ffa70fae25a7e0d755ed55cf223f43913f587db7",
        license="ASL (Academic Software License)",
    ),
    "mace-off23-small": dict(
        url="https://raw.githubusercontent.com/ACEsuit/mace-off/"
            "91a78c5a9c300d1104700d9352c8bfe449227737/mace_off23/MACE-OFF23_small.model",
        file="MACE-OFF23_small.model",
        size=7347350,
        sha256="165cce4cfec5a34b9c64d4ebf95de15d71106bb584b7291c8470f0749977c46f",
        license="ASL (Academic Software License)",
    ),
    # MACE-MP-0 medium and small, release mace_mp_0 of ACEsuit/mace-mp. MIT.
    "mace-mp-0-medium": dict(
        url="https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/"
            "2023-12-03-mace-128-L1_epoch-199.model",
        file="2023-12-03-mace-128-L1_epoch-199.model",
        size=44422970,
        sha256="01bfe22100139f424713cf921144e5509cbe353d67aa9fa1be9c6e1e0ed35845",
        license="MIT",
    ),
    "mace-mp-0-small": dict(
        url="https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/"
            "2023-12-10-mace-128-L0_energy_epoch-249.model",
        file="2023-12-10-mace-128-L0_energy_epoch-249.model",
        size=32581838,
        sha256="2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736",
        license="MIT",
    ),
}
DEFAULT = ["painn-nabladft-100k", "mace-off23-medium"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name, directory):
    a = ARTIFACTS[name]
    path = os.path.join(directory, a["file"])
    if os.path.exists(path) and sha256(path) == a["sha256"]:
        print(f"{name}: {path} present, hash checked")
        return path
    print(f"{name}: downloading {a['url']} ({a['size']} bytes; license {a['license']})")
    tmp = path + ".part"
    with urllib.request.urlopen(a["url"], timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    digest = sha256(tmp)
    if digest != a["sha256"]:
        os.remove(tmp)
        sys.exit(f"{name}: SHA-256 {digest} is not the pinned {a['sha256']}; removed")
    os.replace(tmp, path)
    print(f"{name}: {path}, hash checked")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("dir", nargs="?")
    p.add_argument("names", nargs="*")
    p.add_argument("--list", action="store_true")
    a = p.parse_args()
    if a.list or not a.dir:
        for name, x in ARTIFACTS.items():
            mark = " (default)" if name in DEFAULT else ""
            print(f"{name}{mark}: {x['file']}, {x['size']} bytes, {x['license']}")
        return
    os.makedirs(a.dir, exist_ok=True)
    for name in a.names or DEFAULT:
        if name not in ARTIFACTS:
            sys.exit(f"unknown artifact {name}; --list prints the names")
        fetch(name, a.dir)


if __name__ == "__main__":
    main()
