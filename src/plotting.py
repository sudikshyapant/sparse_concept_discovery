import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def plot_concept_heatmaps(image, S_hat_image, active_idx, splice_encoder, grid_size, top_k=6):
    """LGMD section 3.6: project estimated semantic coefficients back to the spatial
    domain and overlay them on the input image, one heatmap per concept, labeled with
    its SpLiCE vocabulary word.

    image: PIL.Image (original, un-preprocessed)
    S_hat_image: (grid_size*grid_size, r_active) estimated coefficients for this image
        only, in the same row-major (row, then col) order as clip_grid.CLIPGridEncoder.
    active_idx: (r_active,) tensor mapping columns of S_hat_image back to vocab indices.
    splice_encoder: splice_utils.SpliceEncoder, used to look up concept names.
    """
    concept_strength = S_hat_image.sum(dim=0)  # (r_active,)
    top_local = torch.topk(concept_strength, k=min(top_k, concept_strength.shape[0])).indices

    fig, axes = plt.subplots(1, len(top_local), figsize=(4 * len(top_local), 4))
    if len(top_local) == 1:
        axes = [axes]

    img_np = np.array(image)
    for ax, local_idx in zip(axes, top_local):
        heatmap = S_hat_image[:, local_idx].reshape(grid_size, grid_size)
        heatmap = heatmap / (heatmap.max() + 1e-8)
        heatmap_up = F.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0),
            size=(img_np.shape[0], img_np.shape[1]),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()

        concept_name = splice_encoder.concept_name(active_idx[local_idx].item())
        ax.imshow(img_np)
        ax.imshow(heatmap_up, cmap="jet", alpha=0.45)
        ax.set_title(concept_name)
        ax.axis("off")

    fig.tight_layout()
    return fig
