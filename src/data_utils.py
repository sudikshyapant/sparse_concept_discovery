import os
import random
import tarfile

import requests
from PIL import Image
from tqdm.auto import tqdm

from .config import CONFIG


def _download_imagenette():
    """Downloads and extracts imagenette2-160 into CONFIG['data_dir'] if not already present."""
    extracted_dir = os.path.join(CONFIG["data_dir"], "imagenette2-160")
    if os.path.isdir(extracted_dir):
        return extracted_dir

    tgz_path = os.path.join(CONFIG["data_dir"], "imagenette2-160.tgz")
    if not os.path.isfile(tgz_path):
        resp = requests.get(CONFIG["imagenette_url"], stream=True)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(tgz_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="Downloading imagenette2-160"
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    with tarfile.open(tgz_path) as tar:
        tar.extractall(CONFIG["data_dir"])

    return extracted_dir


def _list_class_images(root, class_id):
    """Collects image paths for a class across the train/ and val/ splits shipped in imagenette2-160."""
    paths = []
    for split in ("train", "val"):
        class_dir = os.path.join(root, split, class_id)
        if os.path.isdir(class_dir):
            paths.extend(
                os.path.join(class_dir, fname)
                for fname in sorted(os.listdir(class_dir))
                if fname.lower().endswith((".jpg", ".jpeg", ".png"))
            )
    return paths


def load_class_splits(seed=0):
    """Downloads Imagenette and returns {class_id: {"train": [PIL.Image,...], "val": [...]}}
    for the classes configured in CONFIG, using CONFIG['n_train_per_class'] /
    CONFIG['n_val_per_class'] images per class (train = factorization set,
    val = held-out inference/evaluation set, matching LGMD's train vs. inference split).
    """
    root = _download_imagenette()
    rng = random.Random(seed)

    splits = {}
    n_train = CONFIG["n_train_per_class"]
    n_val = CONFIG["n_val_per_class"]

    for class_id in CONFIG["classes"]:
        paths = _list_class_images(root, class_id)
        if len(paths) < n_train + n_val:
            raise RuntimeError(
                f"Class {class_id} only has {len(paths)} images, "
                f"need {n_train + n_val} (n_train_per_class + n_val_per_class)."
            )
        rng.shuffle(paths)
        train_paths = paths[:n_train]
        val_paths = paths[n_train : n_train + n_val]

        splits[class_id] = {
            "train": [Image.open(p).convert("RGB") for p in train_paths],
            "val": [Image.open(p).convert("RGB") for p in val_paths],
        }

    return splits
