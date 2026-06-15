import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
CATS_RAW = ROOT / "data" / "raw" / "cats"
PROCESSED = ROOT / "data" / "processed"
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


def center_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def load_and_preprocess(path: Path, size: int) -> np.ndarray | None:
    try:
        img = Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None

    if img.width < 32 or img.height < 32:
        return None

    img = center_crop(img)
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = arr / 127.5 - 1.0
    arr = arr.transpose(2, 0, 1)
    return arr


def collect_images(input_dirs: list[Path], limit: int | None) -> list[Path]:
    paths: list[Path] = []
    for input_dir in input_dirs:
        if not input_dir.exists():
            print(f"[WARN] Skipping missing directory: {input_dir}")
            continue
        paths.extend(
            p for p in input_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        )
    paths = sorted(paths)
    if limit is not None:
        paths = paths[:limit]
    return paths


def save_preview(data: np.ndarray, out_path: Path, n: int = 64) -> None:
    try:
        import math
        import torch
        import torchvision.utils as vutils

        samples = torch.from_numpy(data[:n])
        grid = vutils.make_grid(
            samples,
            nrow=int(math.sqrt(n)),
            normalize=True,
            value_range=(-1, 1),
        )
        grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        preview_path = out_path.with_name(out_path.stem + "_preview.png")
        Image.fromarray(grid_np).save(preview_path)
        print(f"  [OK] Preview saved: {preview_path.relative_to(ROOT)}")
    except ImportError:
        print("  [WARN] torchvision is unavailable; skipping preview.")


def print_report(data: np.ndarray, out_path: Path, skipped: int) -> None:
    n, c, h, w = data.shape
    size_mb = data.nbytes / 1024 ** 2
    print()
    print("=" * 52)
    print("  Preprocessing report")
    print("=" * 52)
    print(f"  Valid images  : {n:>6}")
    print(f"  Skipped images: {skipped:>6}")
    print(f"  Tensor shape  : {data.shape}  (N, C, H, W)")
    print(f"  Dtype         : {data.dtype}")
    print(f"  Value range   : [{data.min():.2f}, {data.max():.2f}]")
    print(f"  File size     : {size_mb:.1f} MB")
    print(f"  Saved to      : {display_path(out_path)}")
    print("=" * 52)


def display_path(path: Path) -> Path:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess image folders into an NCHW .npy tensor")
    parser.add_argument("--size", type=int, default=64, help="Target image size")
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=[CATS_RAW],
        help="One or more raw image directories. Directories are scanned recursively.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npy path. Default: data/processed/cats_<size>.npy",
    )
    parser.add_argument("--preview", action="store_true", help="Save a sample preview grid")
    parser.add_argument("--limit", type=int, default=None, help="Optional image limit for smoke tests")
    args = parser.parse_args()

    out_path = args.output or (PROCESSED / f"cats_{args.size}.npy")
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    paths = collect_images(args.input, args.limit)
    if not paths:
        print(f"[ERROR] No images found in: {', '.join(str(p) for p in args.input)}")
        sys.exit(1)

    print(f"\n[Preprocess] {len(paths)} images -> {args.size}x{args.size}")
    print("  Sources:")
    for input_dir in args.input:
        try:
            shown = input_dir.relative_to(ROOT)
        except ValueError:
            shown = input_dir
        print(f"    - {shown}")
    print(f"  Output: {display_path(out_path)}\n")

    results = []
    skipped = 0
    for path in tqdm(paths, desc="Processing", unit="img"):
        arr = load_and_preprocess(path, args.size)
        if arr is None:
            skipped += 1
        else:
            results.append(arr)

    if not results:
        print("[ERROR] No image passed filtering.")
        sys.exit(1)

    data = np.stack(results, axis=0)
    np.save(out_path, data)
    print_report(data, out_path, skipped)
    if args.preview:
        save_preview(data, out_path)


if __name__ == "__main__":
    main()
