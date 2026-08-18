"""
SecMI_stat attack (Duan et al. 2023). Ported directly from their confirmed
source (github.com/jinhaoduan/SecMI, mia_evals/secmia.py:get_intermediate_results
+ naive_statistic_attack), NOT reconstructed from paper-text equations, since an
initial reconstruction attempt from secondary academic sources turned out to
disagree with their actual code on the round-trip indexing. We match the real
implementation exactly:

Given clean image x0, t_sec, and stride k (their CLI default: t_sec=100, k=10):
    target_steps = [k, 2k, 3k, ..., largest multiple of k strictly less than t_sec]
    (i.e. range(0, t_sec, k)[1:] -- note this means the actual "inverted" position
    ends at target_steps[-1], which can be slightly less than t_sec if t_sec isn't
    itself a multiple of k -- this is a real quirk of their code, not a bug we're
    introducing)

    1. x_sec = ddim_multistep(x0, t_c=0, target_steps)  -- k-strided DDIM inversion
    2. probe forward:  x_sec -> (target_steps[-1] -> target_steps[-1]+k), CONSISTENT labels
    3. probe backward: that result -> (target_steps[-1]+k -> target_steps[-1]), CONSISTENT labels
    4. t-error = || x_sec - x_sec_recon ||^2  (squared L2, summed over all pixels)

No "same-t mismatch" trick is actually used (unlike what secondary sources
suggested) -- both probe steps use straightforward, consistent (t_c, t_target)
label pairs matching each step's true position in the schedule.
Smaller t-error => more member-like.
"""
import torch

from attacks.common import ddim_singlestep, ddim_multistep


@torch.no_grad()
def secmi_statistic(diffusion, x0, t_sec, k, device):
    """
    x0: (B, C, H, W). t_sec, k: python ints (SecMI's own defaults: t_sec=100, k=10).
    Returns (B,) tensor of per-image squared-L2 t-error statistics.
    """
    target_steps = list(range(0, t_sec, k))[1:]
    if len(target_steps) == 0:
        raise ValueError(f"t_sec={t_sec}, k={k} gives no inversion steps "
                          f"(need t_sec > k). Use SecMI's defaults (t_sec=100, k=10) "
                          f"or increase t_sec / decrease k.")

    x_sec = ddim_multistep(diffusion, x0, t_c=0, target_steps=target_steps, device=device)
    x_sec = x_sec["x_t_target"]

    last = target_steps[-1]
    step_fwd = ddim_singlestep(diffusion, x_sec, t_c=last, t_target=last + k, device=device)
    step_back = ddim_singlestep(diffusion, step_fwd["x_t_target"], t_c=last + k, t_target=last, device=device)
    x_sec_recon = step_back["x_t_target"]

    return (x_sec - x_sec_recon).flatten(1).pow(2).sum(dim=1)