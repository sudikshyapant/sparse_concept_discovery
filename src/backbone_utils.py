import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from tqdm.auto import tqdm

from .config import CONFIG

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_PREPROCESS = transforms.Compose(
    [
        transforms.Resize((CONFIG["input_size"], CONFIG["input_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class Backbone:
    """Wraps a pretrained ResNet18, exposing the encoder f(.) (spatial activations
    at CONFIG['layer']) and the classifier head g(.) = avgpool + fc, matching the
    f / g decomposition from LGMD section 3.1.
    """

    def __init__(self, device=_DEVICE):
        self.device = device
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        model.eval().to(device)
        self.model = model

        self._activations = None
        layer = getattr(model, CONFIG["layer"])
        layer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self._activations = out

    @torch.no_grad()
    def extract_activations(self, images, batch_size=16):
        """images: list of PIL.Image -> Z: tensor (n, p, h, w)"""
        batches = []
        for i in tqdm(range(0, len(images), batch_size), desc="ResNet18 activations", leave=False):
            batch = images[i : i + batch_size]
            x = torch.stack([_PREPROCESS(img) for img in batch]).to(self.device)
            self.model(x)
            batches.append(self._activations.detach().cpu())
        return torch.cat(batches, dim=0)

    @torch.no_grad()
    def gap_and_classify(self, Z):
        """Z: (n, p, h, w) -> logits (n, num_classes), i.e. g(GAP(Z))."""
        Z = Z.to(self.device)
        pooled = self.model.avgpool(Z).flatten(1)
        logits = self.model.fc(pooled)
        return logits.cpu()


def downsample_to_grid(Z, grid_size):
    """Average-pools spatial activations Z (n, p, h, w) down to (n, p, grid_size, grid_size)
    so that Abar's rows align 1:1 with the CLIP grid-crop rows (see clip_grid.py).
    """
    return F.adaptive_avg_pool2d(Z, (grid_size, grid_size))


def unfold(Z):
    """Z: (n, p, h, w) -> Abar: (n*h*w, p), row-major over (n, h, w)."""
    n, p, h, w = Z.shape
    return Z.permute(0, 2, 3, 1).reshape(n * h * w, p)


def fold(Abar, n, h, w):
    """Inverse of unfold: (n*h*w, p) -> (n, p, h, w)."""
    p = Abar.shape[1]
    return Abar.reshape(n, h, w, p).permute(0, 3, 1, 2)
