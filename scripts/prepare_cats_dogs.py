import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from scripts.preprocess_data import collect_images, load_and_preprocess, print_report, save_preview


DATA_RAW = ROOT / "data" / "raw"
CATS_DIR = DATA_RAW / "cats"
DOGS_DIR = DATA_RAW / "dogs"
TMP_DIR = ROOT / "data" / "_tmp_dogs_vs_cats"
PROCESSED = ROOT / "data" / "processed"


def extract_zip(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(dest)


def find_dog_images(path: Path) -> list[Path]:
    return sorted(
        p for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.name.lower().startswith("dog")
    )


def prepare_from_zip(zip_path: Path) -> Path:
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True)

    extract_zip(zip_path, TMP_DIR)
    for nested_zip in list(TMP_DIR.rglob("*.zip")):
        nested_dest = nested_zip.with_suffix("")
        nested_dest.mkdir(parents=True, exist_ok=True)
        extract_zip(nested_zip, nested_dest)
    return TMP_DIR


def download_competition_zip() -> Path:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        "dogs-vs-cats",
        "-p",
        str(TMP_DIR),
    ]
    subprocess.run(cmd, check=True)
    candidates = sorted(TMP_DIR.glob("*.zip"))
    if not candidates:
        raise FileNotFoundError("Kaggle download completed but no zip file was found")
    return candidates[0]


def copy_dogs(source_root: Path, limit: int | None) -> int:
    dogs = find_dog_images(source_root)
    if limit is not None:
        dogs = dogs[:limit]
    if not dogs:
        raise FileNotFoundError(f"No dog images found under {source_root}")

    DOGS_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in tqdm(dogs, desc="Copying dogs", unit="img"):
        dest = DOGS_DIR / src.name
        if dest.exists():
            continue
        shutil.copy2(src, dest)
        copied += 1
    return copied


def build_combined_npy(size: int, output: Path, limit_per_class: int | None, preview: bool) -> None:
    if not CATS_DIR.exists():
        raise FileNotFoundError(f"Cat directory not found: {CATS_DIR}")
    if not DOGS_DIR.exists():
        raise FileNotFoundError(f"Dog directory not found: {DOGS_DIR}")

    cat_paths = collect_images([CATS_DIR], limit_per_class)
    dog_paths = collect_images([DOGS_DIR], limit_per_class)
    paths = sorted(cat_paths + dog_paths)
    if not paths:
        raise FileNotFoundError("No cat or dog images found")

    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    skipped = 0
    for path in tqdm(paths, desc="Preprocessing cats+dogs", unit="img"):
        arr = load_and_preprocess(path, size)
        if arr is None:
            skipped += 1
        else:
            results.append(arr)

    data = np.stack(results, axis=0)
    np.save(output, data)
    print_report(data, output, skipped)
    if preview:
        save_preview(data, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the exploratory cats+dogs dataset")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dogs-zip", type=Path, help="Path to dogs-vs-cats.zip or train.zip")
    source.add_argument("--dogs-dir", type=Path, help="Already extracted Dogs vs Cats directory")
    source.add_argument("--download", action="store_true", help="Download dogs-vs-cats with the Kaggle CLI")
    parser.add_argument("--dog-limit", type=int, default=None, help="Optional dog image limit")
    parser.add_argument("--build-npy", action="store_true", help="Also build data/processed/cats_dogs_64.npy")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--limit-per-class", type=int, default=None, help="Limit cats and dogs for the combined .npy")
    parser.add_argument("--output", type=Path, default=PROCESSED / "cats_dogs_64.npy")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--keep-temp", action="store_true", help="Keep extracted temporary files")
    args = parser.parse_args()
    if args.dogs_zip is not None and not args.dogs_zip.is_absolute():
        args.dogs_zip = ROOT / args.dogs_zip
    if args.dogs_dir is not None and not args.dogs_dir.is_absolute():
        args.dogs_dir = ROOT / args.dogs_dir
    if not args.output.is_absolute():
        args.output = ROOT / args.output

    uses_temp = False
    if args.download:
        zip_path = download_competition_zip()
        source_root = prepare_from_zip(zip_path)
        uses_temp = True
    elif args.dogs_zip:
        source_root = prepare_from_zip(args.dogs_zip)
        uses_temp = True
    elif args.dogs_dir:
        source_root = args.dogs_dir
    else:
        source_root = DOGS_DIR

    if source_root != DOGS_DIR:
        copied = copy_dogs(source_root, args.dog_limit)
        print(f"Copied {copied} new dog images to {DOGS_DIR.relative_to(ROOT)}")
    else:
        print(f"Using existing dog directory: {DOGS_DIR.relative_to(ROOT)}")

    if args.build_npy:
        build_combined_npy(args.size, args.output, args.limit_per_class, args.preview)
        print(f"Combined dataset saved to {args.output.relative_to(ROOT)}")
    else:
        print("Raw dogs are ready. Add --build-npy to create the combined training tensor.")

    if uses_temp and not args.keep_temp:
        shutil.rmtree(TMP_DIR, ignore_errors=True)
        print(f"Removed temporary extraction directory: {TMP_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
