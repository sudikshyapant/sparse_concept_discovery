import sys

import torch

from .config import CONFIG

_SPLICE_REPO = f"{CONFIG['third_party_dir']}/splice"
if _SPLICE_REPO not in sys.path:
    sys.path.insert(0, _SPLICE_REPO)

import splice as splice_lib  # noqa: E402  (vendored AI4LIFE-GROUP/SpLiCE, third_party/splice)

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SpliceEncoder:
    """Thin wrapper around the vendored official SpLiCE solver. Loads the released
    OpenCLIP ViT-B/32 + LAION-vocabulary dictionary, then exposes encode() to decompose
    *already-computed* CLIP embeddings (e.g. from clip_grid.CLIPGridEncoder) into the
    sparse, non-negative, named-concept coefficient matrix S used as LGMD's fixed
    semantic coefficient matrix (see LGMD section 3.2/3.3).
    """

    def __init__(self, device=_DEVICE):
        self.device = device
        # download_root: SpLiCE caches the vocab list, mean vectors, and (slowest to
        # produce) the per-word CLIP text embeddings here. Pointed at
        # CONFIG["external_cache_dir"] (Drive-backed in Colab) so this is computed
        # once and reused across sessions instead of recomputing on every fresh
        # Colab runtime.
        self.model = splice_lib.load(
            CONFIG["clip_model"],
            vocabulary=CONFIG["splice_vocab"],
            vocabulary_size=CONFIG["splice_vocab_size"],
            l1_penalty=CONFIG["splice_l1_penalty"],
            solver=CONFIG["splice_solver"],
            return_weights=True,
            return_cosine=False,
            device=device,
            download_root=CONFIG["external_cache_dir"] + "/splice",
        )
        # We feed pre-computed CLIP grid embeddings, not raw images, so disable the
        # internal clip.encode_image() call path inside SPLICE.encode_image().
        self.model.clip = None
        self.model.eval()

        self.vocab = splice_lib.get_vocabulary(
            CONFIG["splice_vocab"],
            CONFIG["splice_vocab_size"],
            download_root=CONFIG["external_cache_dir"] + "/splice",
        )

    @torch.no_grad()
    def encode(self, clip_embeddings):
        """clip_embeddings: (n, d) L2-normalized CLIP embeddings -> S: (n, vocab_size)
        non-negative sparse weights.
        """
        return self.model.encode_image(clip_embeddings.to(self.device)).cpu()

    def concept_name(self, concept_idx):
        return self.vocab[concept_idx]
