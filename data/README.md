# Data Directory

This directory contains datasets used in the Cat Image Synthesis project.

## Structure

```
data/
├── raw/
│   ├── cats/          # Raw cat images from Kaggle Dogs vs. Cats dataset
│   └── dogs/          # Raw dog images (for Experiment V)
└── processed/
    ├── cats_64/       # Preprocessed cat images (64x64)
    └── cats_dogs_64/  # Preprocessed cats+dogs images (64x64, for Exp. V)
```

## Datasets

### Primary: Cats (from Dogs vs. Cats)
- **Source:** Kaggle — `dogs-vs-cats`
- **Command:** `kaggle competitions download -c dogs-vs-cats`
- ~12,500 cat images

### Secondary: AFHQ (Animal Faces HQ) — optional upgrade
- **Source:** Kaggle — `andrewmvd/animal-faces`
- High-quality 512x512 animal faces (cats, dogs, wildlife)
- **Command:** `kaggle datasets download -d andrewmvd/animal-faces`

## Download Script

Run `scripts/download_data.py` to automatically download and organize all datasets.
