"""
Evaluation for the Guided Diffusion / ImageNet-1k experiment -- matches
SimA's ACTUAL INv2_attack.py (confirmed from their real source): class-
conditional checkpoint, stratified n-per-class sampling, Resize(BICUBIC)+
CenterCrop preprocessing, labels passed to every query.

Usage:
    python evaluate_imagenet.py \
        --checkpoint models/256x256_diffusion.pt \
        --imagenet1k_root ./data/imagenet1k \
        --imagenetv2_root ./data/imagenetv2_extracted
"""
import argparse
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, ".")

from guided_diffusion_adapter import load_guided_diffusion
from imagenet_data import load_imagenet_member_heldout, sample_per_class, _base_transform

import torchvision

from attacks.loss_attack import loss_statistic
from attacks.pia import pia_statistic
from attacks.sima import sima_statistic, sima_mc_statistic
from attacks.secmi import secmi_statistic
from attacks.common import batched


def tpr_at_fpr(labels, scores, target_fpr=0.01):
    fpr, tpr, _ = roc_curve(labels, scores)
    mask = fpr <= target_fpr
    if not np.any(mask):
        return 0.0
    return float(tpr[mask].max())


def best_balanced_accuracy(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    acc = (tpr * n_pos + (1 - fpr) * n_neg) / len(labels)
    return float(acc.max())


def summarize(labels, raw_stat, higher_raw_means_member):
    scores = raw_stat if higher_raw_means_member else -raw_stat
    return {
        "auc": roc_auc_score(labels, scores),
        "asr": best_balanced_accuracy(labels, scores),
        "tpr@1%fpr": tpr_at_fpr(labels, scores, 0.01),
    }


def compute_stat_batched_labeled(fn, images, labels, model_wrapper, batch_size, device):
    """For q=1 methods only -- no internal repeat_interleave, 1:1 image:label."""
    out = []
    for start in range(0, images.shape[0], batch_size):
        end = start + batch_size
        img_batch = images[start:end].to(device)
        label_batch = labels[start:end].to(device)
        model_wrapper.current_labels = label_batch
        out.append(fn(x0=img_batch).cpu())
    model_wrapper.current_labels = None
    return torch.cat(out)


def compute_stat_batched_labeled_mc(fn, images, labels, model_wrapper, batch_size, device, q):
    """For SimA-MC specifically -- sima_mc_statistic loops internally over q
    draws, calling query_eps with the ORIGINAL (unexpanded) batch size on
    EACH iteration, not one single repeat_interleave-expanded call. So
    current_labels should NOT be repeated here -- unlike LRT+Shape-MC's own
    compute_norms_per_radius_labeled, which genuinely does one expanded call
    and needs repeated labels to match. These are two different internal
    batching strategies; q is accepted for API consistency but unused."""
    out = []
    for start in range(0, images.shape[0], batch_size):
        end = start + batch_size
        img_batch = images[start:end].to(device)
        label_batch = labels[start:end].to(device)
        model_wrapper.current_labels = label_batch  # NOT repeated
        out.append(fn(x0=img_batch).cpu())
    model_wrapper.current_labels = None
    return torch.cat(out)


# --- LRT+Shape (level + slope) helpers ---

def compute_norms_per_radius_labeled(diffusion, x0, labels, t, radius_grid, model_wrapper,
                                       device, batch_size, q_ref=1):
    """q_ref=1: 1:1 image:label. q_ref>1: labels repeated to match the
    internal per-radius q_ref averaging."""
    a_t = diffusion.sqrt_alphas_bar[t].item()
    sigma_t = diffusion.sqrt_one_minus_alphas_bar[t].item()
    b = x0.shape[0]
    norms = np.zeros((b, len(radius_grid)))
    for ri, ratio in enumerate(radius_grid):
        x0_rep = x0.repeat_interleave(q_ref, dim=0)
        labels_rep = labels.repeat_interleave(q_ref, dim=0)
        noise = torch.randn(x0_rep.shape) * (ratio * sigma_t)
        g_full = a_t * x0_rep + noise
        col = []
        for start in range(0, g_full.shape[0], batch_size):
            end = start + batch_size
            g_batch = g_full[start:end].to(device)
            label_batch = labels_rep[start:end].to(device)
            t_batch = torch.full((g_batch.shape[0],), t, dtype=torch.long, device=device)
            model_wrapper.current_labels = label_batch
            with torch.no_grad():
                eps = diffusion.query_eps(g_batch, t_batch)
            col.append(eps.flatten(1).norm(dim=1).cpu())
        norms_flat = torch.cat(col)  # (b*q_ref,)
        norms[:, ri] = norms_flat.view(b, q_ref).mean(dim=1).numpy()
    model_wrapper.current_labels = None
    return norms


def slope_between(norms, radius_grid, i_low, i_high):
    r = np.array(radius_grid)
    return (norms[:, i_high] - norms[:, i_low]) / (r[i_high] - r[i_low])


def grid_search_level_and_direction(norms_calib, labels_calib, radius_grid, n_angles):
    slope_calib = slope_between(norms_calib, radius_grid, 0, len(radius_grid) - 1)
    best_i_level, best_theta, best_auc, best_stats = None, None, -1, None
    for i_level in range(len(radius_grid)):
        level_calib = norms_calib[:, i_level]
        mean_l, std_l = level_calib.mean(), level_calib.std() + 1e-8
        mean_s, std_s = slope_calib.mean(), slope_calib.std() + 1e-8
        level_std = (level_calib - mean_l) / std_l
        slope_std = (slope_calib - mean_s) / std_s
        for theta in np.linspace(0, 2 * np.pi, n_angles, endpoint=False):
            score = np.cos(theta) * level_std + np.sin(theta) * slope_std
            auc = roc_auc_score(labels_calib, score)
            if auc > best_auc:
                best_auc, best_i_level, best_theta = auc, i_level, theta
                best_stats = (mean_l, std_l, mean_s, std_s)
    return best_i_level, best_theta, best_auc, best_stats


def apply_level_and_direction(norms_eval, radius_grid, i_level, theta, stats):
    mean_l, std_l, mean_s, std_s = stats
    level_eval = norms_eval[:, i_level]
    slope_eval = slope_between(norms_eval, radius_grid, 0, len(radius_grid) - 1)
    level_std = (level_eval - mean_l) / std_l
    slope_std = (slope_eval - mean_s) / std_s
    return np.cos(theta) * level_std + np.sin(theta) * slope_std


def run_shape_method(label, member_calib, member_calib_labels, heldout_calib, heldout_calib_labels,
                      member_eval, member_eval_labels, heldout_eval, heldout_eval_labels,
                      labels_calib, labels_eval, diffusion, model_wrapper, device, args, q_ref):
    t0 = time.time()
    best_overall = {"auc": -1, "t": None, "i_level": None, "theta": None, "stats": None}
    for t in args.shape_t_grid:
        norms_calib_m = compute_norms_per_radius_labeled(
            diffusion, member_calib, member_calib_labels, t, args.shape_radius_grid,
            model_wrapper, device, args.batch_size, q_ref)
        norms_calib_h = compute_norms_per_radius_labeled(
            diffusion, heldout_calib, heldout_calib_labels, t, args.shape_radius_grid,
            model_wrapper, device, args.batch_size, q_ref)
        norms_calib = np.concatenate([norms_calib_m, norms_calib_h], axis=0)
        i_level, theta, auc, stats = grid_search_level_and_direction(
            norms_calib, labels_calib, args.shape_radius_grid, args.shape_n_angles)
        if auc > best_overall["auc"]:
            best_overall = {"auc": auc, "t": t, "i_level": i_level, "theta": theta, "stats": stats}
    print(f"  [{label}] calibrated t={best_overall['t']}, "
          f"level_radius={args.shape_radius_grid[best_overall['i_level']]}, "
          f"theta={best_overall['theta']:.3f} (val AUC={best_overall['auc']:.4f})")

    best_t = best_overall["t"]
    norms_eval_m = compute_norms_per_radius_labeled(
        diffusion, member_eval, member_eval_labels, best_t, args.shape_radius_grid,
        model_wrapper, device, args.batch_size, q_ref)
    norms_eval_h = compute_norms_per_radius_labeled(
        diffusion, heldout_eval, heldout_eval_labels, best_t, args.shape_radius_grid,
        model_wrapper, device, args.batch_size, q_ref)
    norms_eval = np.concatenate([norms_eval_m, norms_eval_h], axis=0)
    scores_eval = apply_level_and_direction(norms_eval, args.shape_radius_grid,
                                             best_overall["i_level"], best_overall["theta"],
                                             best_overall["stats"])
    metrics = {
        "auc": roc_auc_score(labels_eval, scores_eval),
        "asr": best_balanced_accuracy(labels_eval, scores_eval),
        "tpr@1%fpr": tpr_at_fpr(labels_eval, scores_eval, 0.01),
    }
    n_q = len(args.shape_radius_grid) * q_ref
    print(f"  [{label}] TEST: AUC={metrics['auc']:.4f} ASR={metrics['asr']:.4f} "
          f"TPR@1%FPR={metrics['tpr@1%fpr']:.4f}  (#Query={n_q})")
    print(f"  ({time.time()-t0:.1f}s)")
    return metrics, {"t": best_t, "radius": args.shape_radius_grid[best_overall["i_level"]],
                      "n_query": n_q}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--imagenet1k_root", type=str, required=True)
    ap.add_argument("--imagenetv2_root", type=str, required=True)
    ap.add_argument("--n_per_class", type=int, default=3)
    ap.add_argument("--n_shadow_per_class", type=int, default=1)
    ap.add_argument("--t_grid", type=int, nargs="+", default=list(range(0, 200, 10)))
    ap.add_argument("--secmi_k", type=int, default=15,
                     help="moderate compromise: standard default is 10 (expensive at this "
                          "scale), earlier cost-cut used 20 (likely too coarse, near-chance "
                          "result). 15 splits the difference.")
    ap.add_argument("--secmi_t_grid", type=int, nargs="+",
                     default=[20, 50, 80, 110, 140, 170, 190],
                     help="moderately larger than the earlier 4-value cost-cut grid, "
                          "short of the full 10-value dense grid")
    ap.add_argument("--sima_mc_t_grid", type=int, nargs="+", default=[10, 50, 90, 130, 190],
                     help="dedicated, smaller grid for SimA-MC (was inheriting the full "
                          "20-value --t_grid, making mc=30 cost ~31h alone). Matches "
                          "Shape's 5-value grid size for a fairer cost comparison.")
    ap.add_argument("--sima_mc_grid", type=int, nargs="+", default=[10, 30],
                     help="matches SimA's own Table 3 rows (#mc=10, #mc=30)")
    ap.add_argument("--lrt_t_grid", type=int, nargs="+", default=[10, 90, 190])
    ap.add_argument("--lrt_alpha_grid", type=float, nargs="+", default=[0.02, 0.06])
    ap.add_argument("--shape_t_grid", type=int, nargs="+", default=[10, 50, 90, 130, 190])
    ap.add_argument("--shape_radius_grid", type=float, nargs="+", default=[0.05, 0.2, 0.4, 0.6])
    ap.add_argument("--shape_n_angles", type=int, default=16)
    ap.add_argument("--shape_mc_q_ref", type=int, default=3,
                     help="multi-query tier for LRT+Shape-MC, matching the spirit of "
                          "SimA-MC's q=10/30 rows -- kept modest (3) given each q_ref "
                          "multiplies cost by (radii x q_ref) at this resolution")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    print("Loading CLASS-CONDITIONAL guided-diffusion checkpoint...")
    diffusion = load_guided_diffusion(args.checkpoint, device=device, T=args.T)
    model_wrapper = diffusion.model
    print("Loaded.")

    member_images, member_labels, heldout_images, heldout_labels = load_imagenet_member_heldout(
        args.imagenet1k_root, args.imagenetv2_root, n_per_class=args.n_per_class,
        seed=args.seed)

    import random as _random
    shadow_rng = _random.Random(args.seed + 1)
    tx = _base_transform(256)
    imn1k_full = torchvision.datasets.ImageFolder(str(args.imagenet1k_root), transform=tx)
    shadow_idx = sample_per_class(imn1k_full, n=args.n_shadow_per_class, rng=shadow_rng)
    shadow_images = torch.stack([imn1k_full[i][0] for i in shadow_idx[:500]])
    print(f"Shadow set: {shadow_images.shape[0]} images")

    def split_val_test(n):
        perm = rng.permutation(n)
        half = n // 2
        return perm[:half], perm[half:]

    m_calib_idx, m_eval_idx = split_val_test(len(member_images))
    h_calib_idx, h_eval_idx = split_val_test(len(heldout_images))
    member_calib, member_calib_labels = member_images[m_calib_idx], member_labels[m_calib_idx]
    member_eval, member_eval_labels = member_images[m_eval_idx], member_labels[m_eval_idx]
    heldout_calib, heldout_calib_labels = heldout_images[h_calib_idx], heldout_labels[h_calib_idx]
    heldout_eval, heldout_eval_labels = heldout_images[h_eval_idx], heldout_labels[h_eval_idx]
    print(f"Validation: {len(member_calib)} member / {len(heldout_calib)} held-out")
    print(f"Test:       {len(member_eval)} member / {len(heldout_eval)} held-out")

    labels_calib = np.concatenate([np.ones(len(member_calib)), np.zeros(len(heldout_calib))])
    labels_eval = np.concatenate([np.ones(len(member_eval)), np.zeros(len(heldout_eval))])

    results = {}

    def calibrate_and_eval(method_label, stat_fn, higher_raw_means_member, t_candidates,
                            batched_fn=compute_stat_batched_labeled, q=1):
        best_t, best_auc = None, -1
        for t in t_candidates:
            s_m = batched_fn(lambda x0, t=t: stat_fn(x0, t), member_calib,
                              member_calib_labels, model_wrapper, args.batch_size, device,
                              *([q] if batched_fn is compute_stat_batched_labeled_mc else []))
            s_h = batched_fn(lambda x0, t=t: stat_fn(x0, t), heldout_calib,
                              heldout_calib_labels, model_wrapper, args.batch_size, device,
                              *([q] if batched_fn is compute_stat_batched_labeled_mc else []))
            raw = torch.cat([s_m, s_h]).numpy()
            scores = raw if higher_raw_means_member else -raw
            auc = roc_auc_score(labels_calib, scores)
            if auc > best_auc:
                best_auc, best_t = auc, t
        print(f"  [{method_label}] calibrated t={best_t} (val AUC={best_auc:.4f})")

        s_m = batched_fn(lambda x0: stat_fn(x0, best_t), member_eval,
                          member_eval_labels, model_wrapper, args.batch_size, device,
                          *([q] if batched_fn is compute_stat_batched_labeled_mc else []))
        s_h = batched_fn(lambda x0: stat_fn(x0, best_t), heldout_eval,
                          heldout_eval_labels, model_wrapper, args.batch_size, device,
                          *([q] if batched_fn is compute_stat_batched_labeled_mc else []))
        raw = torch.cat([s_m, s_h]).numpy()
        metrics = summarize(labels_eval, raw, higher_raw_means_member)
        results[method_label] = (metrics, {"t": best_t})
        print(f"  [{method_label}] TEST: AUC={metrics['auc']:.4f} ASR={metrics['asr']:.4f} "
              f"TPR@1%FPR={metrics['tpr@1%fpr']:.4f}")

    print("\n=== Loss ===")
    t0 = time.time()
    calibrate_and_eval("Loss", lambda x0, t: loss_statistic(diffusion, x0, t, device),
                        higher_raw_means_member=False, t_candidates=args.t_grid)
    print(f"  ({time.time()-t0:.1f}s)")

    print("\n=== PIA ===")
    t0 = time.time()
    calibrate_and_eval("PIA", lambda x0, t: pia_statistic(diffusion, x0, t, device),
                        higher_raw_means_member=False, t_candidates=args.t_grid)
    print(f"  ({time.time()-t0:.1f}s)")

    print("\n=== SimA (l4) ===")
    t0 = time.time()
    calibrate_and_eval("SimA (l4)", lambda x0, t: sima_statistic(diffusion, x0, t, device, p=4),
                        higher_raw_means_member=False, t_candidates=args.t_grid)
    print(f"  ({time.time()-t0:.1f}s)")

    for mc in args.sima_mc_grid:
        print(f"\n=== SimA-MC (l4, mc={mc}) ===")
        t0 = time.time()
        calibrate_and_eval(
            f"SimA-MC (l4, mc={mc})",
            lambda x0, t, mc=mc: sima_mc_statistic(diffusion, x0, t, device, n_mc=mc, p=4),
            higher_raw_means_member=False, t_candidates=args.sima_mc_t_grid,
            batched_fn=compute_stat_batched_labeled_mc, q=mc)
        print(f"  ({time.time()-t0:.1f}s)")

    print("\n=== SecMI_stat ===")
    t0 = time.time()
    secmi_candidates = [t for t in args.secmi_t_grid if t > args.secmi_k]
    calibrate_and_eval(
        "SecMI_stat",
        lambda x0, t: secmi_statistic(diffusion, x0, t_sec=t, k=args.secmi_k, device=device),
        higher_raw_means_member=False, t_candidates=secmi_candidates)
    print(f"  ({time.time()-t0:.1f}s)")


    print("\n" + "=" * 70)
    print("Dataset: ImageNet-1k (member) / ImageNetV2 (held-out), "
          "Guided Diffusion 256x256 CLASS-CONDITIONAL")
    print(f"{'Method':<28} {'ASR':>8} {'AUC':>8} {'TPR@1%FPR':>10} {'chosen hp'}")
    print("-" * 70)
    for label, (m, hp) in results.items():
        print(f"{label:<28} {m['asr']*100:>7.2f}% {m['auc']*100:>7.2f}% "
              f"{m['tpr@1%fpr']*100:>9.2f}%  {hp}")
    print("=" * 70)


if __name__ == "__main__":
    main()
