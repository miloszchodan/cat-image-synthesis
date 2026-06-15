# Data Directory

This directory stores local datasets used by the Cat Image Synthesis project. Raw data and processed tensors are ignored by git.

## Structure

```text
data/
  raw/
    cats/          # Crawford cat dataset, usually CAT_00 ... CAT_06 subfolders
    dogs/          # Dogs from Kaggle Dogs vs Cats, for the exploratory extension
  processed/
    cats_64.npy       # Cat-only 64x64 tensor, shape (N, 3, 64, 64), values [-1, 1]
    cats_dogs_64.npy  # Combined cats+dogs tensor for the exploratory experiment
```

## Primary Dataset

- Source: Kaggle `crawford/cat-dataset`
- Task: train generative models on cat images.
- Local target: `data/raw/cats/`

Preprocess:

```powershell
python scripts/preprocess_data.py --input data/raw/cats --output data/processed/cats_64.npy --preview
```

## Exploratory Cats vs Dogs Dataset

- Source: Kaggle competition `dogs-vs-cats`
- Task: combine dogs with the cat dataset and retrain one model from scratch.
- Local target: `data/raw/dogs/`

Prepare and preprocess:

```powershell
python scripts/prepare_cats_dogs.py --dogs-zip path\to\dogs-vs-cats.zip --build-npy --preview
```

This extension is intentionally exploratory. The report should discuss whether generated samples remain class-distinct or become cat/dog hybrids.
