# LGMD-on-SpLiCE: Small-Scale Proof of Concept

Reimplements [LGMD](https://arxiv.org/abs/2412.06093) (Language-Guided Matrix Decomposition) using [SpLiCE](https://arxiv.org/abs/2402.10376) instead of LGMD's original per-class LLM vocabulary + CLIP occlusion. The idea: factorize a CNN's spatial activations `A ≈ S Wᵀ`, where `S` is a fixed, sparse, non-negative, language-grounded coefficient matrix from SpLiCE, and `W` is the learned concept basis.

Runs end-to-end on 3 Imagenette classes with a pretrained ResNet18 backbone — sized to fit Google Colab Pro's 8GB RAM tier.

## How it works

1. **Encoder activations** — ResNet18 `layer4` spatial features for each image (`backbone_utils.py`)
2. **Semantic coefficients (S)** — crop each image into a 4x4 grid, encode each cell with CLIP, decompose with SpLiCE into sparse named-concept weights (`clip_grid.py`, `splice_utils.py`)
3. **Learn the concept basis (W)** — fix `S`, learn `W` via projected gradient descent so `S Wᵀ` reconstructs the activations (`lgmd.py`)
4. **Inference** — for held-out images, estimate `Ŝ` from the learned `W` via non-negative least squares, reconstruct, and compare predictions before/after (`evaluate.py`)
5. **Visualize** — overlay top concept activations as heatmaps on the image, labeled with their SpLiCE words (`plotting.py`)

## Running

Open `poc.ipynb` in Google Colab and run all cells top to bottom. First run downloads Imagenette, ResNet18, and the open_clip checkpoint, and computes SpLiCE's vocabulary embeddings (slow once, then cached).

To run locally instead:

```bash
pip install -r requirements.txt
jupyter notebook poc.ipynb
```


## Output

Results are cached under `cache/` (Drive-backed when run in Colab):
- `cache/results/metrics.csv` — per-class reconstruction error and accuracy
- `cache/figures/` — concept heatmap images
- `cache/external/` — third-party model/vocab caches (so re-running doesn't redownload/recompute everything)

Raw training data (Imagenette) is *not* cached to Drive — it downloads fresh each Colab session to its local ephemeral disk.
