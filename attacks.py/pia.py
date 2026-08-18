"""
PIA attack (Kong et al. 2023), Eq. 20 in the SimA paper:
    PIA(x*, t) = || eps_theta(x*, t=0) - eps_theta(sqrt(abar_t)x* + sqrt(1-abar_t)*eps_theta(x*,0), t) ||
Uses the model's own t=0 prediction as a deterministic "noise" substitute (hence
"Proximal Initialization"), avoiding the extra variance of a random eps draw.
2 NFEs total (t=0 query, then the t query) -- matches the paper's Table 7
"PIA: UNet NFE=2" exactly. Smaller PIA => more member-like.
"""
import torch


@torch.no_grad()
def pia_statistic(diffusion, x0, t, device):
    """x0: (B, C, H, W). t: python int timestep >= 1 (t=0 is the eps source itself)."""
    b = x0.shape[0]
    t0_batch = torch.zeros(b, dtype=torch.long, device=device)
    eps0 = diffusion.query_eps(x0, t0_batch)  # eps_theta(x*, t=0), used AS the noise

    abar_t = diffusion.alphas_bar[t].item()
    x_t = (abar_t ** 0.5) * x0 + (1 - abar_t) ** 0.5 * eps0

    t_batch = torch.full((b,), t, dtype=torch.long, device=device)
    eps_pred_t = diffusion.query_eps(x_t, t_batch)
    return (eps0 - eps_pred_t).flatten(1).norm(dim=1)