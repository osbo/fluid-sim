"""Hutchinson probe matrix Z: same preprocessing as training (Jacobi low-pass)."""

from collections.abc import Callable

import torch

from .config import HUTCHINSON_PROBE_JACOBI_OMEGA, HUTCHINSON_PROBE_JACOBI_STEPS


def jacobi_smooth_hutchinson_z_inplace(
    Z: torch.Tensor,
    jacobi_inv_diag_batched: torch.Tensor,
    n_orig_list: list[int],
    max_n_pad: int,
    az_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    omega: float = HUTCHINSON_PROBE_JACOBI_OMEGA,
    num_steps: int = HUTCHINSON_PROBE_JACOBI_STEPS,
) -> None:
    """
    In-place damped-Jacobi smoothing on probe columns (matches ``train_leaf_only``).

    ``Z``: (B, n_pad, K); ``jacobi_inv_diag_batched``: (B, n_pad); ``az_fn(Z)`` returns ``A @ Z``
    with the same shape as ``Z``.
    """
    with torch.no_grad():
        for _ in range(int(num_steps)):
            AZ_smooth = az_fn(Z)
            Z.sub_(float(omega) * jacobi_inv_diag_batched.unsqueeze(-1) * AZ_smooth)
        for b_idx, n_o in enumerate(n_orig_list):
            if n_o < max_n_pad:
                Z[b_idx, n_o:, :].zero_()
