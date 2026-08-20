"""
SimA / SimA-MC (the paper this whole project is replicating/comparing against).

Paper finds l4 norm performs best in general (Fig. 3 ablation); default p=4
here, l2 available via the p argument for direct comparison.
Smaller A => more member-like (Eq. 12's decision rule).
"""
import torch


@torch.no_grad()
def sima_statistic(diffusion, x0, t, device, p=4):
    """
    Single-query SimA, matching the OFFICIAL CODE's actual behavior: queries
    the model directly on the raw clean image x0 (no noise, no scaling),
    labeled with timestep t. x0: (B,C,H,W). Returns (B,) tensor.
    """
    b = x0.shape[0]
    t_batch = torch.full((b,), t, dtype=torch.long, device=device)
    eps_pred = diffusion.query_eps(x0, t_batch)
    return eps_pred.flatten(1).norm(p=p, dim=1)


@torch.no_grad()
def sima_mc_statistic(diffusion, x0, t, device, n_mc=30, p=4, rng=None):
    """
    SimA-MC, matching the OFFICIAL CODE's actual aggregation order: average the
    noise-prediction VECTORS over n_mc noisy draws first, THEN take one norm of
    the averaged vector -- not the average of n_mc individually-computed norms
    (that was our original, incorrect implementation; see module docstring).
    x_t = sqrt(abar_t)x0 + sqrt(1-abar_t)*eps -- properly noised each draw,
    matching get_xt(x0, step, eps) in their SimA_MC.ddim_reverse.
    """
    b = x0.shape[0]
    abar_t = diffusion.alphas_bar[t].item()
    t_batch = torch.full((b,), t, dtype=torch.long, device=device)
    accum = torch.zeros_like(x0)  # accumulate VECTORS, not norms
    for _ in range(n_mc):
        eps = torch.randn(x0.shape, device=device, generator=rng)
        x_t = (abar_t ** 0.5) * x0 + (1 - abar_t) ** 0.5 * eps
        eps_pred = diffusion.query_eps(x_t, t_batch)
        accum += eps_pred
    eps_avg = accum / n_mc
    return eps_avg.flatten(1).norm(p=p, dim=1)  # ONE norm, of the averaged vector
