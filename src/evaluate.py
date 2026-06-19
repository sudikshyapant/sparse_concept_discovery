import torch

from . import backbone_utils


def reconstruction_error(Abar, Abar_hat):
    """Frobenius-norm reconstruction error, normalized by Abar's own norm so it's
    comparable across classes/runs.
    """
    return (torch.linalg.matrix_norm(Abar - Abar_hat) / torch.linalg.matrix_norm(Abar)).item()


def predictive_preservation(backbone: backbone_utils.Backbone, Z, Abar_hat, target_class_idx):
    """Compares the backbone's predictions on original activations Z vs. reconstructed
    activations Abar_hat (LGMD section 3.5: predictive preservation is evaluated by
    comparing classification accuracy/agreement between original and reconstructed
    activations).

    Z: (n, p, h, w) original encoder activations.
    Abar_hat: (n*h*w, p) reconstructed unfolded activations, same (n, h, w) as Z.
    target_class_idx: ImageNet class index these images belong to (for accuracy).

    Returns dict with original/reconstructed top-1 accuracy and prediction agreement.
    """
    n, p, h, w = Z.shape
    Z_hat = backbone_utils.fold(Abar_hat, n, h, w)

    logits_orig = backbone.gap_and_classify(Z)
    logits_recon = backbone.gap_and_classify(Z_hat)

    preds_orig = logits_orig.argmax(dim=1)
    preds_recon = logits_recon.argmax(dim=1)

    return {
        "orig_accuracy": (preds_orig == target_class_idx).float().mean().item(),
        "recon_accuracy": (preds_recon == target_class_idx).float().mean().item(),
        "agreement": (preds_orig == preds_recon).float().mean().item(),
    }
