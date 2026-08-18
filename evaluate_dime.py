"""
DIME (Denoiser Ideal Membership Error) evaluation for CIFAR-10, CIFAR-100,
STL10-U, CelebA (the shared ddpm/UNet pipeline). Hutchinson-trace divergence
+ Monte-Carlo-averaged level, q+1 total queries per image per (t,gamma)
evaluated.

Usage:
    python evaluate_dime.py \
        --checkpoint checkpoints/secmi_cifar10.pt --dataset cifar10 \
        --data_root ./data --split_path ./data/member_split.npz
"""
import argparse
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, ".")
from dime_core import grid_search_level_divergence, apply_level_divergence, apply_level_only, apply_M

from ddpm.model import UNet
from ddpm.diffusion import GaussianDiffusion
from checkpoint_loader import load_checkpoint_flexible
from data import get_member_heldout_loaders, get_train_split_noaug


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
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--split_path", type=str, required=True)
    ap.add_argument("--n_calib", type=int, default=500)
    ap.add_argument("--n_eval", type=int, default=5000)
    ap.add_argument("--q_ref_grid", type=int, nargs="+", default=[1, 10, 30],
                     help="perturbed-query count; ACTUAL total query cost per image is "
                          "q_ref+1 (1 shared base query needed for divergence)")
    ap.add_argument("--t_grid", type=int, nargs="+",
                     default=[50, 70, 90, 110, 130, 150, 170, 190, 210, 230])
    ap.add_argument("--gamma_grid", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.4, 0.6])
    ap.add_argument("--n_angles", type=int, default=16)
    ap.add_argument("--ch", type=int, default=128)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=128,
                     help="physical batch size for GPU queries")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)  # seeds z_k perturbation draws too, not just
    # image selection -- needed for run-to-run reproducibility, especially
    # at small n_calib where unseeded perturbation noise can dominate.

    model = UNet(T=args.T, ch=args.ch).to(device)
    load_checkpoint_flexible(args.checkpoint, model, device)
    model.eval()
    diffusion = GaussianDiffusion(model, T=args.T, device=device)

    def query_fn_factory(t):
        def query_fn(g_batch, aux_slice):  # aux_slice unused -- unconditional model
            t_batch = torch.full((g_batch.shape[0],), t, dtype=torch.long, device=device)
            return diffusion.query_eps(g_batch, t_batch)
        return query_fn

    def sqrt_abar_t_fn(t):
        return diffusion.sqrt_alphas_bar[t].item()

    def sigma_t_fn(t):
        return diffusion.sqrt_one_minus_alphas_bar[t].item()

    _, _, member_idx, heldout_idx = get_member_heldout_loaders(
        args.data_root, args.split_path, dataset=args.dataset, batch_size=1,
        num_workers=0, for_training=False)
    base = get_train_split_noaug(args.data_root, dataset=args.dataset)

    def split_val_test(idx_array):
        perm = rng.permutation(len(idx_array))
        half = len(idx_array) // 2
        return np.array(idx_array)[perm[:half]], np.array(idx_array)[perm[half:]]

    m_calib_idx, m_eval_idx = split_val_test(member_idx)
    h_calib_idx, h_eval_idx = split_val_test(heldout_idx)
    m_calib_idx = rng.choice(m_calib_idx, size=min(args.n_calib, len(m_calib_idx)), replace=False)
    h_calib_idx = rng.choice(h_calib_idx, size=min(args.n_calib, len(h_calib_idx)), replace=False)
    m_eval_idx = rng.choice(m_eval_idx, size=min(args.n_eval, len(m_eval_idx)), replace=False)
    h_eval_idx = rng.choice(h_eval_idx, size=min(args.n_eval, len(h_eval_idx)), replace=False)

    member_calib = torch.stack([base[i][0] for i in m_calib_idx])
    heldout_calib = torch.stack([base[i][0] for i in h_calib_idx])
    member_eval = torch.stack([base[i][0] for i in m_eval_idx])
    heldout_eval = torch.stack([base[i][0] for i in h_eval_idx])

    labels_calib = np.concatenate([np.ones(len(member_calib)), np.zeros(len(heldout_calib))])
    labels_eval = np.concatenate([np.ones(len(member_eval)), np.zeros(len(heldout_eval))])
    x0_calib = torch.cat([member_calib, heldout_calib], dim=0)
    x0_eval = torch.cat([member_eval, heldout_eval], dim=0)

    print(f"Validation: {len(member_calib)} member / {len(heldout_calib)} held-out")
    print(f"Test:       {len(member_eval)} member / {len(heldout_eval)} held-out")

    results = {}
    for q_ref in args.q_ref_grid:
        total_query = q_ref + 1
        print(f"\n=== q_ref={q_ref} (total queries={total_query}) ===")
        t0 = time.time()
        best_level, best_combined, best_M = grid_search_level_divergence(
            query_fn_factory, x0_calib, sqrt_abar_t_fn, sigma_t_fn,
            args.t_grid, args.gamma_grid, q_ref, labels_calib, device,
            args.batch_size, args.n_angles)

        # --- Level alone ---
        print(f"  [Level]    calibrated t={best_level['t']}, gamma={best_level['gamma']} "
              f"(val AUC={best_level['auc']:.4f})")
        query_fn_l = query_fn_factory(best_level["t"])
        sqrt_abar_t_l = sqrt_abar_t_fn(best_level["t"])
        sigma_t_l = sigma_t_fn(best_level["t"])
        level_scores, _ = apply_level_only(
            query_fn_l, x0_eval, sqrt_abar_t_l, sigma_t_l, best_level["gamma"], q_ref,
            best_level["mean"], best_level["std"], best_level["sign"], device, args.batch_size)
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
            best_combined["theta"], best_combined["stats"], device, args.batch_size)
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
            best_M["mean"], best_M["std"], best_M["sign"], device, args.batch_size)
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
    print(f"Dataset: {args.dataset}  (DIME)")
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