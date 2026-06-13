import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
ROOT       = Path(__file__).resolve().parent.parent
CATS_RAW   = ROOT / "data" / "raw" / "cats"
PROCESSED  = ROOT / "data" / "processed"
def center_crop(img: Image.Image) -> Image.Image:
    w, h  = img.size
    side  = min(w, h)
    left  = (w - side) // 2
    top   = (h - side) // 2
    return img.crop((left, top, left + side, top + side))
def load_and_preprocess(path: Path, size: int) -> np.ndarray | None:
    try:
        img = Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None
    if img.width < 32 or img.height < 32:
        return None
    img  = center_crop(img)
    img  = img.resize((size, size), Image.LANCZOS)
    arr  = np.array(img, dtype=np.float32)      
    arr  = arr / 127.5 - 1.0                    
    arr  = arr.transpose(2, 0, 1)               
    return arr
def save_preview(data: np.ndarray, out_path: Path, n: int = 64) -> None:
    try:
        import math
        import torchvision.utils as vutils
        import torch
        samples = torch.from_numpy(data[:n])           
        grid    = vutils.make_grid(samples, nrow=int(math.sqrt(n)),
                                   normalize=True, value_range=(-1, 1))
        grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        preview_path = out_path.with_name(out_path.stem + "_preview.png")
        Image.fromarray(grid_np).save(preview_path)
        print(f"  [✓] Preview zapisany → {preview_path.relative_to(ROOT)}")
    except ImportError:
        print("  [!] torchvision niedostępne — pomijam preview.")
def print_report(data: np.ndarray, out_path: Path, skipped: int) -> None:
    n, c, h, w = data.shape
    size_mb     = data.nbytes / 1024 ** 2
    print()
    print("=" * 52)
    print("  Raport preprocessingu")
    print("=" * 52)
    print(f"  Obrazy poprawne : {n:>6}")
    print(f"  Obrazy odrzucone: {skipped:>6}")
    print(f"  Shape tensora   : {data.shape}  (N, C, H, W)")
    print(f"  Dtype           : {data.dtype}")
    print(f"  Zakres wartości : [{data.min():.2f}, {data.max():.2f}]")
    print(f"  Rozmiar pliku   : {size_mb:.1f} MB")
    print(f"  Zapis           : {out_path.relative_to(ROOT)}")
    print("=" * 52)
    vram_estimate_mb = size_mb  
    if vram_estimate_mb < 500:
        print(f"\n  💡 {size_mb:.0f} MB — dataset zmieści się w całości na GPU VRAM.")
        print("     W DataLoaderze użyj:")
        print("       tensor = torch.from_numpy(np.load(...)).cuda()")
    else:
        print(f"\n  ⚠️  {size_mb:.0f} MB — rozważ pin_memory=True w DataLoaderze.")
def main() -> None:
    parser = argparse.ArgumentParser(description="Etap 1: Preprocessing obrazów kotów")
    parser.add_argument("--size",    type=int,  default=64,
                        help="Docelowy rozmiar obrazu (default: 64)")
    parser.add_argument("--input",   type=Path, default=CATS_RAW,
                        help="Katalog z surowymi obrazami")
    parser.add_argument("--output",  type=Path, default=None,
                        help="Ścieżka wyjściowa .npy (default: data/processed/cats_<size>.npy)")
    parser.add_argument("--preview", action="store_true",
                        help="Zapisz siatkę próbek PNG po preprocessingu")
    args = parser.parse_args()
    out_path = args.output or (PROCESSED / f"cats_{args.size}.npy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    extensions = ["*.jpg", "*.jpeg", "*.png"]
    paths = []
    for ext in extensions:
        paths.extend(args.input.glob(ext))
    if not paths:
        print(f"[ERROR] Brak obrazów w: {args.input}")
        print("  → Uruchom najpierw: python scripts/download_data.py")
        sys.exit(1)
    print(f"\n[Etap 1] Preprocessing: {len(paths)} obrazów -> {args.size}x{args.size}")
    print(f"  Źródło  : {args.input.relative_to(ROOT)}")
    print(f"  Cel     : {out_path.relative_to(ROOT)}\n")
    results = []
    skipped = 0
    for p in tqdm(paths, desc="Przetwarzanie", unit="img"):
        arr = load_and_preprocess(p, args.size)
        if arr is None:
            skipped += 1
        else:
            results.append(arr)
    if not results:
        print("[ERROR] Żaden obraz nie przeszedł filtrowania.")
        sys.exit(1)
    data = np.stack(results, axis=0)    
    np.save(out_path, data)
    print_report(data, out_path, skipped)
    if args.preview:
        save_preview(data, out_path)
if __name__ == "__main__":
    main()

