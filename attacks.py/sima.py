"""
SimA / SimA-MC (the paper this whole project is replicating/comparing against).

IMPORTANT: the paper's Eq. 1 states A(x,t) = ||eps_theta(x_t,t)||_p with
x_t = sqrt(abar_t)x* + sqrt(1-abar_t)*eps -- a NOISED query. We initially
implemented it that way. But the actual official code (mx-ethan-rao/SimA,
DDPM/components.py, class SimA.ddim_reverse) queries the model on the RAW,
UNSCALED clean image directly:
    eps = self.eps_getter(x0, condition, self.noise_level, step)   # x0, not x_t!
No noise, no sqrt(abar_t) scaling -- just model(x0, t=step). This is a real
deviation from both the paper's formula AND its own prose (Section 4.2 calls
SimA a "point estimate (eps=0)", which would still imply sqrt(abar_t)*x0, not
raw x0). Confirmed by direct inspection of their repo, not inferred. We match
their actual code here since the goal is reproducing their reported numbers.
Confirmed working: matches the paper's reported ASR/AUC almost exactly on the
real checkpoint (83.60/90.43 vs their 83.69/90.23).

SimA-MC: a SECOND real discrepancy found here, distinct from the above. Their
SimA_MC.ddim_reverse averages the noise-prediction VECTORS across the n_mc
noisy queries FIRST, then the distance() function takes ONE norm of that
averaged vector:
    eps_accum = torch.randn_like(x0)          # (unrelated init quirk, not replicated)
    for _ in range(n_mc):
        eps_accum += eps_theta(get_xt(x0, step, eps), step)   # summing VECTORS
    eps_accum /= n_mc
    # distance() then takes ONE norm of eps_accum
This is NOT what the paper's own Eq. 13 states (sum of norms, i.e. norm computed
per-draw then averaged) -- another code-vs-prose mismatch, and we match the code
since that's what produced their reported numbers. This matters a lot: averaging
VECTORS first lets independent per-draw noise cancel out (the same mechanism
that makes our own LRT's Tbar construction work), which is why their reported
AUC climbs with more MC samples. Averaging norms (our original, wrong,
implementation) doesn't get this cancellation and can actually get WORSE with
more samples -- confirmed: it was scoring below even the single-query SimA.

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