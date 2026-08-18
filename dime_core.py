"""
Core math for DIME (Denoiser Ideal Membership Error) -- architecture-
agnostic. Implements exactly the theory derived in the writeup:

  Level:      hat_L(q) = (1/q) sum_k ||eps*(g0 + gamma*sigma_t*z_k, t)||
  Divergence: hat_D(q) = (1/q) sum_k z_k^T (eps_k - eps_base) / (gamma*sigma_t)

Both use the SAME q perturbed draws z_k (Rademacher probes: z_i in {-1,+1},
chosen over Gaussian for provably lower Hutchinson-estimator variance --
Var(z^T J z) excludes the diagonal-squared term under Rademacher probing.
Satisfies the same E[z]=0, Cov(z)=I_d required by the derivation, so the
theory is unaffected). Total query cost per image, per (t, gamma): q + 1
(q perturbed, shared between level and divergence, + 1 shared base query
at g0).

`query_fn(g_batch, aux_batch)` is a closure that already knows its timestep
t internally; `aux_batch` is an OPTIONAL auxiliary tensor (e.g. class labels
for the conditional pipeline) sliced identically and explicitly alongside
g_batch at every sub-batching point -- avoids any implicit/stateful
position-tracking between calls, which is a known source of subtle,
silent bugs (see: this project's SimA-MC label-repeat incident).
"""
import numpy as np
import torch


def compute_base_query(query_fn, x0_batch, sqrt_abar_t, device, physical_batch_size, aux_batch=None):
    """Computes eps_base = eps*(g0, t) once -- depends only on t (via
    sqrt_abar_t and query_fn), NOT on gamma. Callers sweeping multiple gamma
    values at fixed t should call this ONCE and reuse the result."""
    b = x0_batch.shape[0]
    g0 = sqrt_abar_t * x0_batch
    chunks = []
    for start in range(0, b, physical_batch_size):
        end = start + physical_batch_size
        aux_slice = aux_batch[start:end].to(device) if aux_batch is not None else None
        chunks.append(query_fn(g0[start:end].to(device), aux_slice))
    return torch.cat(chunks, dim=0)


def compute_level_divergence_batch(query_fn, x0_batch, sqrt_abar_t, sigma_t, gamma, q,
                                     device, physical_batch_size, aux_batch=None,
                                     eps_base_precomputed=None):
    """
    x0_batch: (b, C, H, W) CPU tensor.
    aux_batch: optional (b, ...) CPU tensor (e.g. class labels), sliced
        identically to x0_batch/g at every call -- pass None if unused.
    eps_base_precomputed: optional, pass in a base query already computed via
        compute_base_query() to avoid recomputing it (base only depends on t,
        not gamma -- callers sweeping gamma at fixed t should precompute once).
    query_fn(g_batch, aux_slice) -> eps_batch.
    Returns: hat_L, hat_D, each shape (b,) numpy arrays.
    """
    b = x0_batch.shape[0]
    g0 = sqrt_abar_t * x0_batch

    def sliced_query(g_full, start, end):
        aux_slice = aux_batch[start:end].to(device) if aux_batch is not None else None
        return query_fn(g_full[start:end].to(device), aux_slice)

    if eps_base_precomputed is not None:
        eps_base = eps_base_precomputed
    else:
        eps_base_chunks = []
        for start in range(0, b, physical_batch_size):
            end = start + physical_batch_size
            eps_base_chunks.append(sliced_query(g0, start, end))
        eps_base = torch.cat(eps_base_chunks, dim=0)

    # --- q perturbed queries, shared between level and divergence ---
    # Rademacher probes (z_i in {-1,+1}) instead of Gaussian: satisfies the
    # same E[z]=0, Cov(z)=I_d required by the Hutchinson identity (so the
    # derivation is unaffected), but provably reduces estimator variance --
    # Var(z^T J z) for Rademacher excludes the diagonal-squared term present
    # in the Gaussian case (Hutchinson 1989; Avron & Toledo 2011).
    level_sum = torch.zeros(b, device=device)
    div_sum = torch.zeros(b, device=device)
    for k in range(q):
        z_k = (torch.randint(0, 2, g0.shape, dtype=torch.float32) * 2 - 1)  # Rademacher, CPU
        g_k = g0 + gamma * sigma_t * z_k
        eps_k_chunks = []
        for start in range(0, b, physical_batch_size):
            end = start + physical_batch_size
            eps_k_chunks.append(sliced_query(g_k, start, end))
        eps_k = torch.cat(eps_k_chunks, dim=0)

        z_k_dev = z_k.to(device)
        level_sum += eps_k.flatten(1).norm(dim=1)
        div_sum += (z_k_dev.flatten(1) * (eps_k - eps_base).flatten(1)).sum(dim=1) / (gamma * sigma_t)

    hat_L = (level_sum / q).cpu().numpy()
    hat_D = (div_sum / q).cpu().numpy()
    return hat_L, hat_D


def standardize(values, mean=None, std=None):
    if mean is None:
        mean, std = values.mean(), values.std() + 1e-8
    return (values - mean) / std, mean, std


def grid_search_level_divergence(query_fn_factory, x0_calib, sqrt_abar_t_fn, sigma_t_fn,
                                   t_grid, gamma_grid, q, labels_calib, device,
                                   physical_batch_size, n_angles=16, aux_calib=None):
    """
    Computes (hat_L, hat_D) for EVERY (t, gamma) cell once, caching results,
    then does THREE independent post-hoc selections from the same cache:
      - best_level: (t, gamma) maximizing Level (L^2) alone's calibration AUC
      - best_combined: (t, gamma, theta) maximizing the free-theta combined score
      - best_M: (t, gamma) maximizing the theory-fixed estimation-error
        statistic M = sigma_t^2 * L^2 - sigma_t^3 * D, per the
        bias-variance decomposition -- no theta search, weighting fixed by
        theory rather than fit to calibration data.
    No duplicate network queries between the three -- all read the same cache.
    Returns (best_level, best_combined, best_M), each a dict.
    """
    from sklearn.metrics import roc_auc_score
    cache = {}  # (t, gamma) -> (hat_L, hat_D, sqrt_abar_t, sigma_t)
    for t in t_grid:
        query_fn = query_fn_factory(t)
        sqrt_abar_t, sigma_t = sqrt_abar_t_fn(t), sigma_t_fn(t)

        # Base query computed ONCE for this t, reused across every gamma below
        eps_base_full = compute_base_query(query_fn, x0_calib, sqrt_abar_t, device,
                                             physical_batch_size, aux_batch=aux_calib)
        for gamma in gamma_grid:
            hat_L, hat_D = compute_level_divergence_batch(
                query_fn, x0_calib, sqrt_abar_t, sigma_t, gamma, q, device,
                physical_batch_size, aux_batch=aux_calib, eps_base_precomputed=eps_base_full)
            cache[(t, gamma)] = (hat_L, hat_D, sqrt_abar_t, sigma_t)

    # --- Selection 1: Level alone (uses L^2, per the bias-variance decomposition) ---
    # Sign is FIXED by theory (lower response magnitude near x* => more likely
    # member), NOT chosen by comparing calibration AUC vs 1-AUC. That
    # comparison is a noisy binary decision at small n_calib -- confirmed to
    # cause catastrophic sign flips (AUC ~0.30 instead of ~0.70) on tiny
    # (n=25) calibration sets; removing it costs nothing at large n_calib,
    # where the "correct" direction was already being selected reliably.
    best_level = {"auc": -1, "t": None, "gamma": None, "mean": None, "std": None}
    for (t, gamma), (hat_L, hat_D, sqrt_abar_t, sigma_t) in cache.items():
        hat_L2 = hat_L ** 2
        L2_std, mean_l2, std_l2 = standardize(hat_L2)
        score = -L2_std  # fixed sign: lower L^2 => higher (more member-like) score
        auc = roc_auc_score(labels_calib, score)
        if auc > best_level["auc"]:
            best_level = {"auc": auc, "t": t, "gamma": gamma, "mean": mean_l2, "std": std_l2,
                           "sign": -1.0}

    # --- Selection 2: Combined (free-theta, L^2 + divergence) ---
    best_combined = {"auc": -1, "t": None, "gamma": None, "theta": None, "stats": None}
    for (t, gamma), (hat_L, hat_D, sqrt_abar_t, sigma_t) in cache.items():
        hat_L2 = hat_L ** 2
        L2_std, mean_l2, std_l2 = standardize(hat_L2)
        D_std, mean_d, std_d = standardize(hat_D)
        for theta in np.linspace(0, 2 * np.pi, n_angles, endpoint=False):
            score = np.cos(theta) * L2_std + np.sin(theta) * D_std
            auc = roc_auc_score(labels_calib, score)
            if auc > best_combined["auc"]:
                best_combined = {"auc": auc, "t": t, "gamma": gamma, "theta": theta,
                                  "stats": (mean_l2, std_l2, mean_d, std_d)}

    # --- Selection 3: M (theory-fixed weighting, no theta search) ---
    # CORRECTED: weight = sigma_t**3 (no sqrt_abar_t division -- see module
    # docstring for the numerical verification of this correction).
    best_M = {"auc": -1, "t": None, "gamma": None, "mean": None, "std": None}
    for (t, gamma), (hat_L, hat_D, sqrt_abar_t, sigma_t) in cache.items():
        weight = sigma_t ** 3  # CORRECTED: was (sigma_t**3)/sqrt_abar_t
        hat_M = (sigma_t ** 2) * (hat_L ** 2) - weight * hat_D
        M_std, mean_m, std_m = standardize(hat_M)
        auc_raw = roc_auc_score(labels_calib, M_std)
        auc = max(auc_raw, 1 - auc_raw)
        if auc > best_M["auc"]:
            best_M = {"auc": auc, "t": t, "gamma": gamma, "mean": mean_m, "std": std_m,
                      "sign": 1.0 if auc_raw >= 1 - auc_raw else -1.0}

    return best_level, best_combined, best_M


def apply_level_only(query_fn, x0_eval, sqrt_abar_t, sigma_t, gamma, q, mean, std, sign,
                       device, physical_batch_size, aux_eval=None):
    hat_L, _ = compute_level_divergence_batch(
        query_fn, x0_eval, sqrt_abar_t, sigma_t, gamma, q, device,
        physical_batch_size, aux_batch=aux_eval)
    hat_L2 = hat_L ** 2  # bias-variance decomposition uses L^2, not raw L
    L2_std = sign * (hat_L2 - mean) / std
    return L2_std, hat_L


def apply_level_divergence(query_fn, x0_eval, sqrt_abar_t, sigma_t, gamma, q, theta, stats,
                             device, physical_batch_size, aux_eval=None):
    mean_l2, std_l2, mean_d, std_d = stats
    hat_L, hat_D = compute_level_divergence_batch(
        query_fn, x0_eval, sqrt_abar_t, sigma_t, gamma, q, device,
        physical_batch_size, aux_batch=aux_eval)
    hat_L2 = hat_L ** 2
    L2_std = (hat_L2 - mean_l2) / std_l2
    D_std = (hat_D - mean_d) / std_d
    return np.cos(theta) * L2_std + np.sin(theta) * D_std, hat_L, hat_D


def apply_M(query_fn, x0_eval, sqrt_abar_t, sigma_t, gamma, q, mean, std, sign,
            device, physical_batch_size, aux_eval=None):
    hat_L, hat_D = compute_level_divergence_batch(
        query_fn, x0_eval, sqrt_abar_t, sigma_t, gamma, q, device,
        physical_batch_size, aux_batch=aux_eval)
    weight = sigma_t ** 3  # CORRECTED: was (sigma_t**3)/sqrt_abar_t
    hat_M = (sigma_t ** 2) * (hat_L ** 2) - weight * hat_D
    M_std = sign * (hat_M - mean) / std
    return M_std, hat_M