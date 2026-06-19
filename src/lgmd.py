import torch
from tqdm.auto import tqdm


def _active_columns(S):
    """Indices of concept columns that are non-zero for at least one row of S.
    Restricting the factorization to these (instead of all ~10000 vocab columns)
    keeps S^T S well-conditioned and PGD cheap -- inactive columns would otherwise
    correspond to zero columns in S, making S^T S singular along those dimensions.
    """
    active = torch.nonzero(S.sum(dim=0) > 0, as_tuple=True)[0]
    return active


def learn_basis(Abar, S, lr, steps):
    """LGMD eq. 1-2: learn the concept basis W given fixed semantic coefficients S.

    Abar: (m, p) unfolded encoder activations.
    S: (m, r_full) non-negative semantic coefficient matrix (e.g. from SpliceEncoder).

    Returns (W, active_idx) where W: (p, r_active) and active_idx maps each column
    of W back to its index in the original r_full-sized vocabulary.

    W is initialized at zero rather than via a closed-form least-squares solve:
    r_active (number of concepts active anywhere in S) commonly approaches or
    exceeds m (number of training rows = n_images * grid_size^2) in this POC's
    low-data regime, making S^T S rank-deficient. A closed-form inverse there
    blows up along unconstrained directions and overfits training rows at the
    expense of held-out reconstruction. Zero-init + plain PGD (the paper's actual
    Eq. 1-2 optimizer) is implicitly regularized by early stopping and stays
    numerically stable regardless of rank.
    """
    active_idx = _active_columns(S)
    S_active = S[:, active_idx]

    r_active = S_active.shape[1]
    W = torch.zeros(Abar.shape[1], r_active)

    for _ in tqdm(range(steps), desc="PGD learning basis W", leave=False):
        recon = S_active @ W.T  # (m, p)
        resid = recon - Abar
        grad = resid.T @ S_active  # (p, r_active)
        W = (W - lr * grad).clamp(min=0)

    return W, active_idx


def estimate_coefficients(Abar_new, W, refine_steps=50, ridge=1e-4):
    """LGMD eq. 3-4: estimate non-negative semantic coefficients for new (held-out)
    activations under a fixed, already-learned basis W.

    Abar_new: (m_new, p)
    W: (p, r_active)

    Returns S_hat: (m_new, r_active).
    """
    r_active = W.shape[1]
    gram = W.T @ W + ridge * torch.eye(r_active)
    S_hat = (Abar_new @ W @ torch.linalg.inv(gram)).clamp(min=0)

    if refine_steps > 0:
        eta = 1.0 / (torch.linalg.matrix_norm(gram, ord=2) + 1e-8)
        for _ in tqdm(range(refine_steps), desc="PGD refining S_hat", leave=False):
            grad = S_hat @ gram - Abar_new @ W
            S_hat = (S_hat - eta * grad).clamp(min=0)

    return S_hat


def reconstruct(S_hat, W):
    """LGMD eq. 5: Ahat = S_hat @ W^T."""
    return S_hat @ W.T
