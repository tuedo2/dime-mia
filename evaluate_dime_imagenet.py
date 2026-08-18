"""
DIME (Denoiser Ideal Membership Error) evaluation for ImageNet-1k /
class-conditional Guided Diffusion. Same core/theory as evaluate_dime.py;
class labels are threaded through EXPLICITLY via aux_batch (sliced
identically to the image batch at every call), avoiding any implicit/
stateful position tracking between calls.

Usage:
    python evaluate_dime_imagenet.py \
        --checkpoint models/256x256_diffusion.pt \
        --imagenet1k_root ./data/imagenet1k --imagenetv2_root <path>
"""
import argparse
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, ".")
from dime_core import grid_search_level_divergence, apply_level_divergence, apply_level_only, apply_M

from guided_diffusion_adapter import load_guided_diffusion
from imagenet_data import load_imagenet_member_heldout


def tpr_at_fpr(labels, scores, target_fpr=0.01):
    fpr, tpr, _ = roc_curve(labels, scores)
    mask = fpr <= target_fpr
    return float(tpr[mask].max()) if np.any(mask) else 0.0


def best_balanced_accuracy(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    return float(((tpr * n_pos + (1 - fpr) * n_neg) / len(labels)).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--imagenet1k_root", type=str, required=True)
    ap.add_argument("--imagenetv2_root", type=str, required=True)
    ap.add_argument("--n_per_class", type=int, default=3)
    ap.add_argument("--q_ref_grid", type=int, nargs="+", default=[1, 10, 30],
                     help="ACTUAL total query cost per image is q_ref+1")
    ap.add_argument("--t_grid", type=int, nargs="+", default=[10, 50, 90, 130, 170, 190])
    ap.add_argument("--gamma_grid", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.4, 0.6])
    ap.add_argument("--n_angles", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=24)
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
        args.imagenet1k_root, args.imagenetv2_root, n_per_class=args.n_per_class, seed=args.seed)

    def split_val_test(n):
        perm = rng.permutation(n)
        half = n // 2
        return perm[:half], perm[half:]

    m_calib_idx, m_eval_idx = split_val_test(len(member_images))
    h_calib_idx, h_eval_idx = split_val_test(len(heldout_images))
    member_calib, mcl = member_images[m_calib_idx], member_labels[m_calib_idx]
    member_eval, mel = member_images[m_eval_idx], member_labels[m_eval_idx]
    heldout_calib, hcl = heldout_images[h_calib_idx], heldout_labels[h_calib_idx]
    heldout_eval, hel = heldout_images[h_eval_idx], heldout_labels[h_eval_idx]
    print(f"Validation: {len(member_calib)} member / {len(heldout_calib)} held-out")
    print(f"Test:       {len(member_eval)} member / {len(heldout_eval)} held-out")

    x0_calib = torch.cat([member_calib, heldout_calib], dim=0)
    x0_eval = torch.cat([member_eval, heldout_eval], dim=0)
    aux_calib = torch.cat([mcl, hcl], dim=0)  # class labels, SAME order as x0_calib
    aux_eval = torch.cat([mel, hel], dim=0)
    labels_calib = np.concatenate([np.ones(len(member_calib)), np.zeros(len(heldout_calib))])
    labels_eval = np.concatenate([np.ones(len(member_eval)), np.zeros(len(heldout_eval))])

    def query_fn_factory(t):
        def query_fn(g_batch, aux_slice):
            model_wrapper.current_labels = aux_slice  # explicit, matches g_batch exactly
            t_batch = torch.full((g_batch.shape[0],), t, dtype=torch.long, device=device)
            with torch.no_grad():
                out = diffusion.query_eps(g_batch, t_batch)
            model_wrapper.current_labels = None  # reset immediately after use
            return out
        return query_fn

    def sqrt_abar_t_fn(t):
        return diffusion.sqrt_alphas_bar[t].item()

    def sigma_t_fn(t):
        return diffusion.sqrt_one_minus_alphas_bar[t].item()

    results = {}
    for q_ref in args.q_ref_grid:
        total_query = q_ref + 1
        print(f"\n=== q_ref={q_ref} (total queries={total_query}) ===")
        t0 = time.time()
        best_level, best_combined, best_M = grid_search_level_divergence(
            query_fn_factory, x0_calib, sqrt_abar_t_fn, sigma_t_fn,
            args.t_grid, args.gamma_grid, q_ref, labels_calib, device,
            args.batch_size, args.n_angles, aux_calib=aux_calib)

        # --- Level alone ---
        print(f"  [Level]    calibrated t={best_level['t']}, gamma={best_level['gamma']} "
              f"(val AUC={best_level['auc']:.4f})")
        query_fn_l = query_fn_factory(best_level["t"])
        sqrt_abar_t_l = sqrt_abar_t_fn(best_level["t"])
        sigma_t_l = sigma_t_fn(best_level["t"])
        level_scores, _ = apply_level_only(
            query_fn_l, x0_eval, sqrt_abar_t_l, sigma_t_l, best_level["gamma"], q_ref,
            best_level["mean"], best_level["std"], best_level["sign"], device, args.batch_size,
            aux_eval=aux_eval)
        level_auc = roc_auc_score(labels_eval, level_scores)
        level_asr = best_balanced_accuracy(labels_eval, level_scores)
        level_tpr = tpr_at_fpr(labels_eval, level_scores, 0.01)
        print(f"  [Level]    TEST: AUC={level_auc:.4f} ASR={level_asr:.4f} TPR@1%FPR={level_tpr:.4f}")

        # --- Combined (free-theta) ---
        print(f"  [Combined] calibrated t={best_combined['t']}, gamma={best_combined['gamma']}, "
              f"theta={best_combined['theta']:.3f} (val AUC={best_combined['auc']:.4f})")
        query_fn_c = query_fn_factory(best_combined["t"])
        sqrt_abar_t_c = sqrt_abar_t_fn(best_combined["t"])
        sigma_t_c = sigma_t_fn(best_combined["t"])
        comb_scores, _, _ = apply_level_divergence(
            query_fn_c, x0_eval, sqrt_abar_t_c, sigma_t_c, best_combined["gamma"], q_ref,
            best_combined["theta"], best_combined["stats"], device, args.batch_size,
            aux_eval=aux_eval)
        comb_auc = roc_auc_score(labels_eval, comb_scores)
        comb_asr = best_balanced_accuracy(labels_eval, comb_scores)
        comb_tpr = tpr_at_fpr(labels_eval, comb_scores, 0.01)
        print(f"  [Combined] TEST: AUC={comb_auc:.4f} ASR={comb_asr:.4f} TPR@1%FPR={comb_tpr:.4f}")

        # --- M (theory-fixed weighting, no theta search) ---
        print(f"  [M]        calibrated t={best_M['t']}, gamma={best_M['gamma']} "
              f"(val AUC={best_M['auc']:.4f})")
        query_fn_m = query_fn_factory(best_M["t"])
        sqrt_abar_t_m = sqrt_abar_t_fn(best_M["t"])
        sigma_t_m = sigma_t_fn(best_M["t"])
        M_scores, _ = apply_M(
            query_fn_m, x0_eval, sqrt_abar_t_m, sigma_t_m, best_M["gamma"], q_ref,
            best_M["mean"], best_M["std"], best_M["sign"], device, args.batch_size,
            aux_eval=aux_eval)
        M_auc = roc_auc_score(labels_eval, M_scores)
        M_asr = best_balanced_accuracy(labels_eval, M_scores)
        M_tpr = tpr_at_fpr(labels_eval, M_scores, 0.01)
        print(f"  [M]        TEST: AUC={M_auc:.4f} ASR={M_asr:.4f} TPR@1%FPR={M_tpr:.4f}")
        print(f"  ({time.time()-t0:.1f}s)")

        results[q_ref] = {
            "total_query": total_query,
            "level": (level_auc, level_asr, level_tpr),
            "combined": (comb_auc, comb_asr, comb_tpr),
            "M": (M_auc, M_asr, M_tpr),
        }

    print("\n" + "=" * 130)
    print("Dataset: ImageNet-1k (DIME)")
    print(f"{'q_ref':>6} {'#Query':>8} | {'Level AUC':>10} {'Level ASR':>10} {'Level TPR@1%FPR':>16} "
          f"| {'Combined AUC':>13} {'Combined ASR':>13} {'Combined TPR@1%FPR':>19} "
          f"| {'M AUC':>8} {'M ASR':>8} {'M TPR@1%FPR':>13}")
    print("-" * 130)
    for q_ref, r in results.items():
        l_auc, l_asr, l_tpr = r["level"]
        c_auc, c_asr, c_tpr = r["combined"]
        m_auc, m_asr, m_tpr = r["M"]
        print(f"{q_ref:>6} {r['total_query']:>8} | {l_auc*100:>9.2f}% {l_asr*100:>9.2f}% "
              f"{l_tpr*100:>15.2f}% | {c_auc*100:>12.2f}% {c_asr*100:>12.2f}% {c_tpr*100:>18.2f}% "
              f"| {m_auc*100:>7.2f}% {m_asr*100:>7.2f}% {m_tpr*100:>12.2f}%")
    print("=" * 130)


if __name__ == "__main__":
    main()