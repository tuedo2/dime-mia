"""
Full evaluation harness, all methods

Usage:
    python evaluate.py --checkpoint checkpoints/secmi_cifar10.pt \
        --dataset cifar10 --data_root ./data --split_path ./data/member_split.npz \
        --n_calib 500 --n_eval 12500 --t_grid_step 20
"""
import argparse
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from ddpm.model import UNet
from ddpm.diffusion import GaussianDiffusion
from checkpoint_loader import load_checkpoint_flexible
from data import get_member_heldout_loaders, get_train_split_noaug, get_shadow_split

from attacks.loss_attack import loss_statistic
from attacks.pia import pia_statistic
from attacks.secmi import secmi_statistic
from attacks.sima import sima_statistic, sima_mc_statistic
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


def compute_stat_batched(fn, images, batch_size, device):
    out = []
    for batch in batched(images, batch_size):
        out.append(fn(x0=batch.to(device)).cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--dataset", type=str, default="cifar10")
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--split_path", type=str, required=True)
    ap.add_argument("--n_calib", type=int, default=500)
    ap.add_argument("--n_eval", type=int, default=12500)
    ap.add_argument("--t_grid_step", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--ch", type=int, default=128)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--secmi_k", type=int, default=10)
    ap.add_argument("--sima_p_grid", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--lrt_alpha_grid", type=float, nargs="+",
                     default=[0.01, 0.02, 0.03, 0.05, 0.08, 0.15, 0.25, 0.4, 0.6])
    ap.add_argument("--calibrate_by", type=str, choices=["auc", "tpr"], default="auc",
                     help="AUC is the default/simple choice here -- since we've confirmed "
                          "our standard-convention numbers already match SecMI/PIA, the "
                          "TPR-specific calibration objective (used earlier to investigate "
                          "the LRT q-degradation) is no longer necessary; plain AUC-based "
                          "sweeping (matching evaluate.py's original, simplest protocol) "
                          "is sufficient")
    ap.add_argument("--n_shadow", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    model = UNet(T=args.T, ch=args.ch).to(device)
    key_used = load_checkpoint_flexible(args.checkpoint, model, device)
    model.eval()
    print(f"Loaded checkpoint via key '{key_used}'  calibrate_by={args.calibrate_by}  "
          f"TPR convention=STANDARD (member=1)")
    diffusion = GaussianDiffusion(model, T=args.T, device=device)

    _, _, member_idx, heldout_idx = get_member_heldout_loaders(
        args.data_root, args.split_path, dataset=args.dataset,
        batch_size=1, num_workers=0, for_training=False)
    base = get_train_split_noaug(args.data_root, dataset=args.dataset)

    def split_val_test(idx):
        idx = np.array(idx)
        perm = rng.permutation(len(idx))
        half = len(idx) // 2
        return idx[perm[:half]], idx[perm[half:]]

    member_val_idx, member_test_idx = split_val_test(member_idx)
    heldout_val_idx, heldout_test_idx = split_val_test(heldout_idx)
    print(f"Validation: {len(member_val_idx)} member / {len(heldout_val_idx)} held-out")
    print(f"Test:       {len(member_test_idx)} member / {len(heldout_test_idx)} held-out")

    def load_images(indices, n):
        n = min(n, len(indices))
        chosen = rng.choice(indices, size=n, replace=False)
        return torch.stack([base[i][0] for i in chosen])

    member_calib = load_images(member_val_idx, args.n_calib)
    heldout_calib = load_images(heldout_val_idx, args.n_calib)
    member_eval = load_images(member_test_idx, args.n_eval)
    heldout_eval = load_images(heldout_test_idx, args.n_eval)

    shadow_ds = get_shadow_split(args.data_root, dataset=args.dataset)
    n_shadow = min(args.n_shadow, len(shadow_ds))
    shadow_idx = rng.choice(len(shadow_ds), size=n_shadow, replace=False)
    shadow_images = torch.stack([shadow_ds[i][0] for i in shadow_idx])
    M_shadow = compute_shadow_M(shadow_images)
    print(f"Shadow M (n={n_shadow}): {M_shadow:.4f}")

    t_grid = list(range(10, 301, args.t_grid_step))
    if 300 not in t_grid:
        t_grid.append(300)
    print(f"t calibration grid: {t_grid}")

    labels_calib = np.concatenate([np.ones(len(member_calib)), np.zeros(len(heldout_calib))])
    labels_eval = np.concatenate([np.ones(len(member_eval)), np.zeros(len(heldout_eval))])

    results = {}

    def calibrate_and_eval(method_label, stat_fn, higher_raw_means_member, t_candidates=t_grid):
        best_t, best_score = None, -1
        for t in t_candidates:
            s_m = compute_stat_batched(lambda x0, t=t: stat_fn(x0, t), member_calib,
                                        args.batch_size, device)
            s_h = compute_stat_batched(lambda x0, t=t: stat_fn(x0, t), heldout_calib,
                                        args.batch_size, device)
            raw = torch.cat([s_m, s_h]).numpy()
            scores = raw if higher_raw_means_member else -raw
            score = (roc_auc_score(labels_calib, scores) if args.calibrate_by == "auc"
                     else tpr_at_fpr(labels_calib, scores, 0.01))
            if score > best_score:
                best_score, best_t = score, t
        print(f"  [{method_label}] calibrated t={best_t} (val {args.calibrate_by}={best_score:.4f})")

        s_m = compute_stat_batched(lambda x0: stat_fn(x0, best_t), member_eval,
                                    args.batch_size, device)
        s_h = compute_stat_batched(lambda x0: stat_fn(x0, best_t), heldout_eval,
                                    args.batch_size, device)
        raw = torch.cat([s_m, s_h]).numpy()
        metrics = summarize(labels_eval, raw, higher_raw_means_member)
        results[method_label] = (metrics, {"t": best_t})
        print(f"  [{method_label}] TEST: AUC={metrics['auc']:.4f} ASR={metrics['asr']:.4f} "
              f"TPR@1%FPR={metrics['tpr@1%fpr']:.4f}")

    # --- Loss, PIA, SecMI_stat, SimA/SimA-MC (unchanged from evaluate_tpr_calibrated.py) ---
    print("\n=== Loss ===")
    t0 = time.time()
    calibrate_and_eval("Loss", lambda x0, t: loss_statistic(diffusion, x0, t, device),
                        higher_raw_means_member=False)
    print(f"  ({time.time()-t0:.1f}s)")

    print("\n=== PIA ===")
    t0 = time.time()
    pia_candidates = [t for t in t_grid if t >= 1]
    calibrate_and_eval("PIA", lambda x0, t: pia_statistic(diffusion, x0, t, device),
                        higher_raw_means_member=False, t_candidates=pia_candidates)
    print(f"  ({time.time()-t0:.1f}s)")

    print("\n=== SecMI_stat ===")
    t0 = time.time()
    secmi_candidates = [t for t in t_grid if t > args.secmi_k]
    calibrate_and_eval(
        "SecMI_stat",
        lambda x0, t: secmi_statistic(diffusion, x0, t_sec=t, k=args.secmi_k, device=device),
        higher_raw_means_member=False, t_candidates=secmi_candidates)
    print(f"  ({time.time()-t0:.1f}s)")

    for p in args.sima_p_grid:
        for label, n_mc in [(f"SimA (l{p})", 1), (f"SimA-MC (l{p}, q=10)", 10),
                             (f"SimA-MC (l{p}, q=30)", 30)]:
            print(f"\n=== {label} ===")
            t0 = time.time()
            if n_mc == 1:
                fn = lambda x0, t, p=p: sima_statistic(diffusion, x0, t, device, p=p)
            else:
                fn = lambda x0, t, n_mc=n_mc, p=p: sima_mc_statistic(
                    diffusion, x0, t, device, n_mc=n_mc, p=p)
            calibrate_and_eval(label, fn, higher_raw_means_member=False)
            print(f"  ({time.time()-t0:.1f}s)")


    # --- Final table ---
    import re

    def infer_query_count(label):
        if label == "Loss":
            return 1
        if label == "PIA":
            return 2
        if label == "SecMI_stat":
            return f"~{args.secmi_k}+2"
        m = re.search(r"q=(\d+)", label)
        if m:
            return int(m.group(1))
        if label.startswith("SimA ("):
            return 1
        return "?"

    print("\n" + "=" * 95)
    print(f"Dataset: {args.dataset}  calibrate_by={args.calibrate_by}  "
          f"TPR convention: STANDARD (member=1, matches SecMI/PIA)")
    print(f"{'Method':<28} {'#Query':>8} {'ASR':>8} {'AUC':>8} {'TPR@1%FPR':>10} {'chosen hp'}")
    print("-" * 95)
    for label, (m, hp) in results.items():
        print(f"{label:<28} {str(infer_query_count(label)):>8} {m['asr']*100:>7.2f}% "
              f"{m['auc']*100:>7.2f}% {m['tpr@1%fpr']*100:>9.2f}%  {hp}")
    print("=" * 95)


if __name__ == "__main__":
    main()
