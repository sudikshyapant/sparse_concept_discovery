import os

IN_COLAB = "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ

if IN_COLAB:
    from google.colab import drive

    drive.mount("/content/drive")
    # Drive-backed: only for things worth persisting across runtime resets
    # (pretrained checkpoints, SpLiCE's slow-to-compute vocab embeddings, our own
    # results/visualizations). Reading/writing here goes over network to Drive,
    # so it's slower than local /content disk -- reserved for the expensive stuff.
    _root = "/content/drive/MyDrive/sparse_concept_discovery"
    # Ephemeral local Colab disk: raw training data (Imagenette). Re-downloaded
    # each fresh runtime, but the 160px variant is ~100MB so that's cheap, and
    # local disk is faster for the repeated image reads during a run.
    _data_root = "/content/sparse_concept_discovery_data"
else:
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _data_root = _root

CONFIG = {
    # Imagenette classes to run the per-class LGMD-on-SpLiCE pipeline on.
    # Folder names from the imagenette2-160 release.
    "classes": ["n02979186", "n03417042", "n03445777"],  # cassette player, garbage truck, golf ball
    "class_display_names": {
        "n02979186": "cassette player",
        "n03417042": "garbage truck",
        "n03445777": "golf ball",
    },
    # Standard ImageNet-1k class indices (the ordering torchvision's pretrained
    # ResNet18 fc layer was trained with) for the wnids above. Used to score
    # predictive-preservation accuracy in evaluate.py. If predictive_preservation's
    # orig_accuracy comes out near 0, double check these indices first.
    "wnid_to_imagenet_idx": {
        "n02979186": 482,  # cassette player
        "n03417042": 569,  # garbage truck
        "n03445777": 574,  # golf ball
    },
    "imagenette_url": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
    "n_train_per_class": 40,
    "n_val_per_class": 10,
    # Backbone
    "backbone": "resnet18",
    "layer": "layer4",
    "input_size": 224,
    # CLIP / SpLiCE
    "clip_model": "open_clip:ViT-B-32",
    "splice_vocab": "laion",
    "splice_vocab_size": -1,  # full released vocabulary
    "splice_l1_penalty": 0.20,
    "splice_solver": "skl",
    "grid_size": 4,  # 4x4 grid-crop, also the spatial resolution Abar/S are aligned to
    # LGMD basis learning
    "pgd_lr": 0.05,
    "pgd_steps": 300,
    "r_concepts_to_show": 8,
    # Paths
    "root_dir": _root,
    "third_party_dir": os.path.join(_root, "third_party"),
    # Raw training data (Imagenette). Ephemeral local Colab disk -- see _data_root.
    "data_dir": os.path.join(_data_root, "data"),
    # Everything below is Drive-backed in Colab (persists across runtime resets):
    "cache_dir": os.path.join(_root, "cache"),
    # Where third-party libraries (SpLiCE, torch hub, open_clip/HF hub) cache their
    # own downloads + computed artifacts (e.g. SpLiCE's per-word CLIP text
    # embeddings, which are slow to recompute).
    "external_cache_dir": os.path.join(_root, "cache", "external"),
    # Our own computed results: per-class metrics tables and concept-heatmap figures.
    "results_dir": os.path.join(_root, "cache", "results"),
    "figures_dir": os.path.join(_root, "cache", "figures"),
}

os.makedirs(CONFIG["data_dir"], exist_ok=True)
os.makedirs(CONFIG["cache_dir"], exist_ok=True)
os.makedirs(CONFIG["external_cache_dir"], exist_ok=True)
os.makedirs(CONFIG["results_dir"], exist_ok=True)
os.makedirs(CONFIG["figures_dir"], exist_ok=True)

os.environ["TORCH_HOME"] = os.path.join(CONFIG["external_cache_dir"], "torch")
os.environ["HF_HOME"] = os.path.join(CONFIG["external_cache_dir"], "huggingface")
