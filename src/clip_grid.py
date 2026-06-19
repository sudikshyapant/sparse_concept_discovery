import open_clip
import torch
from tqdm.auto import tqdm

from .config import CONFIG

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Must match the checkpoint SpLiCE's released "open_clip:ViT-B-32" dictionary/means
# were built with (see third_party/splice/splice/splice.py: pretrained='laion2b_s34b_b79k').
_PRETRAINED = "laion2b_s34b_b79k"


class CLIPGridEncoder:
    """Crops each image into a grid_size x grid_size grid of non-overlapping cells,
    resizes each cell to CLIP's input resolution, and encodes all cells with CLIP.
    Row-major cell order (row-by-row, left-to-right) matches backbone_utils.unfold's
    (n, h, w) -> (n*h*w, p) ordering, so Abar and the resulting CLIP grid embeddings
    line up row-for-row once Abar has been downsampled to the same grid_size.
    """

    def __init__(self, device=_DEVICE):
        self.device = device
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=_PRETRAINED, device=device
        )
        model.eval()
        self.model = model
        self.preprocess = preprocess

    def _grid_crops(self, image, grid_size):
        w, h = image.size
        cell_w, cell_h = w / grid_size, h / grid_size
        crops = []
        for row in range(grid_size):
            for col in range(grid_size):
                box = (
                    int(col * cell_w),
                    int(row * cell_h),
                    int((col + 1) * cell_w),
                    int((row + 1) * cell_h),
                )
                crops.append(image.crop(box))
        return crops

    @torch.no_grad()
    def encode_images(self, images, grid_size=None, batch_size=64):
        """images: list of PIL.Image -> (n*grid_size*grid_size, d) L2-normalized CLIP embeddings,
        row-major per image (row, then col), images concatenated in input order.
        """
        grid_size = grid_size or CONFIG["grid_size"]
        all_crops = []
        for image in images:
            all_crops.extend(self._grid_crops(image, grid_size))

        embeddings = []
        for i in tqdm(range(0, len(all_crops), batch_size), desc="CLIP grid-crop encoding", leave=False):
            batch = all_crops[i : i + batch_size]
            x = torch.stack([self.preprocess(crop) for crop in batch]).to(self.device)
            feats = self.model.encode_image(x)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            embeddings.append(feats.cpu())

        return torch.cat(embeddings, dim=0)
