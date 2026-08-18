"""
Shared utilities for attacks that need deterministic DDIM steps (PIA, SecMI).

ddim_singlestep here is a direct port of SecMI's own ddim_singlestep function
(confirmed from their actual source, mia_evals/secmia.py), just using our
GaussianDiffusion's precomputed alpha_bar schedule instead of recomputing betas
inline. Semantics are identical:

    Given x labeled as being at timestep t_c, predict x0 via
        pred_x0 = (x - sqrt(1-abar[t_c]) * eps_theta(x, t_c)) / sqrt(abar[t_c])
    then reconstruct at timestep t_target:
        x_t_target = sqrt(abar[t_target]) * pred_x0 + sqrt(1-abar[t_target]) * eps_theta(x, t_c)

Note eps_theta(x, t_c) is queried ONCE and reused for both the x0-prediction
and the reconstruction (not re-queried at t_target) -- this matches their code
exactly (a single model call per ddim_singlestep, not two).
"""
import torch


def ddim_singlestep(diffusion, x, t_c, t_target, device):
    """
    x: (B, C, H, W). t_c, t_target: python ints (shared across the batch).
    Returns dict with 'x_t_target' and 'epsilon' (the queried eps_theta(x, t_c)),
    matching SecMI's own return structure.
    """
    b = x.shape[0]
    t_c_batch = torch.full((b,), t_c, dtype=torch.long, device=device)
    eps = diffusion.query_eps(x, t_c_batch)

    abar_c = diffusion.alphas_bar[t_c].item()
    abar_target = diffusion.alphas_bar[t_target].item()

    pred_x0 = (x - (1 - abar_c) ** 0.5 * eps) / (abar_c ** 0.5)
    x_t_target = (abar_target ** 0.5) * pred_x0 + (1 - abar_target) ** 0.5 * eps
    return {"x_t_target": x_t_target, "epsilon": eps}


def ddim_multistep(diffusion, x, t_c, target_steps, device):
    """Chains ddim_singlestep across target_steps, updating t_c after each hop.
    Matches SecMI's ddim_multistep exactly."""
    result = None
    for t_target in target_steps:
        result = ddim_singlestep(diffusion, x, t_c, t_target, device)
        x = result["x_t_target"]
        t_c = t_target
    return result


def batched(tensor, batch_size):
    """Yield successive batch_size-sized chunks of a tensor along dim 0."""
    for i in range(0, tensor.shape[0], batch_size):
        yield tensor[i:i + batch_size]