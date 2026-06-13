import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
CATS_DIR = DATA_RAW / "cats"
DOGS_DIR = DATA_RAW / "dogs"
AFHQ_DIR = DATA_RAW / "afhq"
TMP_DIR  = ROOT / "data" / "_tmp"
KAGGLE_BIN = Path(os.environ.get("USERPROFILE", "~")).expanduser() /             "AppData" / "Roaming" / "Python" / "Python311" / "Scripts" / "kaggle.exe"
def run(cmd: str) -> int:
    print(f"  $ {cmd}")
    return os.system(cmd)
def check_kaggle_credentials():
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("\n[ERROR] Kaggle API key not found!")
        print(f"  Expected: {kaggle_json}")
        print("  → Go to https://www.kaggle.com → Settings → API → Create New Token")
        print("  → Place the downloaded kaggle.json in ~/.kaggle/")
        sys.exit(1)
    print(f"[✓] Kaggle credentials found at {kaggle_json}")
def extract_zip(zip_path: Path, dest: Path):
    print(f"  Extracting {zip_path.name} → {dest} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)
    zip_path.unlink()
    print(f"  [✓] Extracted.")
def download_cats_vs_dogs():
    print("\n[1/2] Downloading Dogs vs. Cats dataset ...")
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    ret = run(
        f'"{KAGGLE_BIN}" competitions download -c dogs-vs-cats -p "{TMP_DIR}"'
    )
    if ret != 0:
        print("[WARN] Competition download failed. Trying dataset mirror ...")
        ret = run(
            f'"{KAGGLE_BIN}" datasets download -d salader/dogs-vs-cats -p "{TMP_DIR}"'
        )
    zip_file = TMP_DIR / "dogs-vs-cats.zip"
    alt_zip  = TMP_DIR / "dogs-vs-cats.zip"
    zips = list(TMP_DIR.glob("*.zip"))
    if not zips:
        print("[ERROR] No zip file found after download.")
        sys.exit(1)
    main_zip = zips[0]
    extract_zip(main_zip, TMP_DIR)
    for inner_zip in TMP_DIR.glob("*.zip"):
        extract_zip(inner_zip, TMP_DIR)
    CATS_DIR.mkdir(parents=True, exist_ok=True)
    DOGS_DIR.mkdir(parents=True, exist_ok=True)
    train_dir = TMP_DIR / "train"
    if not train_dir.exists():
        train_dir = TMP_DIR
    moved_cats = moved_dogs = 0
    for img in train_dir.glob("*.jpg"):
        if img.name.startswith("cat."):
            shutil.move(str(img), CATS_DIR / img.name)
            moved_cats += 1
        elif img.name.startswith("dog."):
            shutil.move(str(img), DOGS_DIR / img.name)
            moved_dogs += 1
    print(f"  [✓] Cats: {moved_cats} images  |  Dogs: {moved_dogs} images")
    shutil.rmtree(TMP_DIR, ignore_errors=True)
def download_afhq():
    print("\n[2/2] Downloading Animal Faces HQ (AFHQ) dataset ...")
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    AFHQ_DIR.mkdir(parents=True, exist_ok=True)
    ret = run(
        f'"{KAGGLE_BIN}" datasets download -d andrewmvd/animal-faces -p "{TMP_DIR}"'
    )
    if ret != 0:
        print("[ERROR] AFHQ download failed.")
        return
    zips = list(TMP_DIR.glob("*.zip"))
    if zips:
        extract_zip(zips[0], AFHQ_DIR)
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print(f"  [✓] AFHQ saved to {AFHQ_DIR}")
def print_summary():
    print("\n" + "=" * 55)
    print("  Dataset Summary")
    print("=" * 55)
    for folder in [CATS_DIR, DOGS_DIR, AFHQ_DIR]:
        if folder.exists():
            count = len(list(folder.rglob("*.jpg"))) + len(list(folder.rglob("*.png")))
            print(f"  {folder.relative_to(ROOT):<30}  {count:>6} images")
    print("=" * 55)
    print("\nNext step: run  python scripts/preprocess_data.py")
def main():
    parser = argparse.ArgumentParser(description="Download datasets for Cat Image Synthesis")
    parser.add_argument("--cats-only", action="store_true",
                        help="Download only the Cats dataset (skip AFHQ)")
    parser.add_argument("--afhq", action="store_true",
                        help="Also download the AFHQ high-quality dataset")
    args = parser.parse_args()
    check_kaggle_credentials()
    download_cats_vs_dogs()
    if args.afhq:
        download_afhq()
    elif not args.cats_only:
        ans = input("\nDownload AFHQ (higher-quality 512x512 faces)? [y/N] ").strip().lower()
        if ans == "y":
            download_afhq()
    print_summary()
if __name__ == "__main__":
    main()

