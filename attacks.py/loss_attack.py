"""
Loss attack (Matsumoto et al. 2023), Eq. 18 in the SimA paper:
    Loss(x*, t) = || eps - eps_theta(sqrt(abar_t) x* + sqrt(1-abar_t) eps, t) ||
where eps ~ N(0,I) is drawn once per call. Smaller Loss => more member-like
(the network reconstructs the actual injected noise better for training points).
"""
import torch


@torch.no_grad()
def loss_statistic(diffusion, x0, t, device, rng=None):
    """
    x0: (B, C, H, W) clean images. t: python int timestep.
    Returns (B,) tensor of per-image L2 statistics.
    """
    b = x0.shape[0]
    eps = torch.randn(x0.shape, device=device, generator=rng)
    abar_t = diffusion.alphas_bar[t].item()
    x_t = (abar_t ** 0.5) * x0 + (1 - abar_t) ** 0.5 * eps

    t_batch = torch.full((b,), t, dtype=torch.long, device=device)
    eps_pred = diffusion.query_eps(x_t, t_batch)
    return (eps - eps_pred).flatten(1).norm(dim=1)