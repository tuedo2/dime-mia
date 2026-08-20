"""
Train a DDPM on a dataset's member split, matching standard Ho et al. (2020)
hyperparameters (as used by SecMI/PIA/SimA for their CIFAR-10/100 DDPMs):

    base channels=128, ch_mult=[1,2,2,2], 2 ResBlocks/stage, attention @ 16x16,
    dropout=0.1, T=1000, linear beta schedule [1e-4, 0.02], batch_size=128,
    Adam lr=2e-4, EMA decay=0.9999, grad clip=1.0.

Now dataset-agnostic (CIFAR-10 / CIFAR-100) via --dataset, matching the
generalization already applied to evaluate.py/quick_eval_check.py. See data.py's
DATASET_REGISTRY for what's supported.

Usage:
    # Step 1: measure real throughput on your actual node (few minutes):
    python train.py --data_root ./data --logdir ./logs/cifar10_ddpm \
        --split_path ./data/member_split.npz --benchmark_only --benchmark_steps 200

    # Step 2: full run, once you know your steps/sec and have picked total_steps:
    python train.py --data_root ./data --logdir ./logs/cifar10_ddpm \
        --split_path ./data/member_split.npz --total_steps 800000

    # Deliberate small-subset overfit run (see make_overfit_split.py to create
    # the small split file first):
    python train.py --data_root ./data --logdir ./logs/cifar10_overfit_n2500 \
        --dataset cifar10 --split_path ./data/overfit_split_n2500.npz \
        --total_steps 50000
"""
import argparse
import copy
import os
import time

import torch
import torch.optim as optim
from torchvision.utils import save_image

from ddpm.model import UNet
from ddpm.diffusion import GaussianDiffusion
from data import get_member_heldout_loaders


def ema_update(ema_model, model, decay):
    with torch.no_grad():
        ema_params = dict(ema_model.named_parameters())
        model_params = dict(model.named_parameters())
        for name, param in model_params.items():
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--dataset", type=str, default="cifar10",
                    help="dataset name, must be registered in data.py's DATASET_REGISTRY")
    p.add_argument("--split_path", type=str, default="./data/member_split.npz")
    p.add_argument("--logdir", type=str, default="./logs/cifar10_ddpm")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--total_steps", type=int, default=50_000,
                    help="default sized for a ~6h single-A40 budget at an estimated "
                         "2-2.6 steps/sec; run --benchmark_only on your actual node "
                         "first and adjust this to match your measured throughput")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=0,
                    help="linear LR warmup over this many steps, ramping from 0 to --lr "
                         "-- matches SimA's own reference DDPM/main.py (warmup=5000 there). "
                         "0 disables warmup (this project's original default, fine at "
                         "batch_size=128; recommended >0, e.g. 5000, when using --parallel "
                         "with a larger batch_size -- jumping straight to full LR at large "
                         "batch size is a known destabilization pattern their own code "
                         "explicitly guards against and this project's original version did "
                         "not implement).")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--beta_1", type=float, default=1e-4)
    p.add_argument("--beta_T", type=float, default=0.02)
    p.add_argument("--ch", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--sample_every", type=int, default=2500)
    p.add_argument("--ckpt_every", type=int, default=2500,
                    help="frequent enough that a SLURM walltime cutoff doesn't lose much progress")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--resume", type=str, default=None,
                    help="path to a checkpoint to resume from")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", action="store_true", default=True,
                    help="use bf16 mixed precision (default on; pass --no-amp to disable)")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--channels_last", action="store_true", default=True)
    p.add_argument("--compile", action="store_true", default=False,
                    help="torch.compile the model (adds first-call compile overhead;"
                         " beneficial for long runs, skews short benchmarks)")
    p.add_argument("--parallel", action="store_true", default=False,
                    help="wrap the model in nn.DataParallel to use multiple GPUs on this "
                         "node (matches SimA's own --parallel flag naming/behavior). "
                         "Splits --batch_size across visible GPUs -- e.g. batch_size=1024 "
                         "on 4 GPUs processes 256/GPU, avoiding the single-GPU OOM seen "
                         "at batch_size=1024 on one A40. Use this to match SimA's stated "
                         "training step counts directly (batch_size=1024) rather than "
                         "scaling total_steps to compensate for a smaller single-GPU batch.")
    p.add_argument("--benchmark_only", action="store_true",
                    help="run --benchmark_steps steps, report real steps/sec and "
                         "projected wall-clock for --total_steps, then exit (no checkpoint saved)")
    p.add_argument("--benchmark_steps", type=int, default=200)
    args = p.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(os.path.join(args.logdir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(args.logdir, "ckpts"), exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    member_loader, _, member_idx, heldout_idx = get_member_heldout_loaders(
        args.data_root, args.split_path, dataset=args.dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, for_training=True)
    print(f"dataset={args.dataset}  member set: {len(member_idx)} images, "
          f"held-out set: {len(heldout_idx)} images")
    data_iter = infinite_loader(member_loader)

    model = UNet(T=args.T, ch=args.ch).to(device)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    # EMA deepcopy happens BEFORE DataParallel wrapping -- nn.DataParallel
    # prefixes parameter names with "module." (since it stores the real model
    # as self.module), which would silently break ema_update()'s direct
    # name-matching between model.named_parameters() and
    # ema_model.named_parameters() if ema_model were copied from an
    # already-wrapped model, or if the wrapped model were passed to
    # ema_update() directly. ema_model itself is NEVER wrapped in
    # DataParallel (no need -- it's not used for training-time forward/
    # backward passes here, only EMA-averaged inference).
    ema_model = copy.deepcopy(model).to(device)
    for p_ in ema_model.parameters():
        p_.requires_grad_(False)

    if args.parallel:
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            print(f"--parallel: wrapping model in nn.DataParallel across {n_gpus} GPUs "
                  f"(batch_size={args.batch_size} split ~{args.batch_size // n_gpus}/GPU)")
            model = torch.nn.DataParallel(model)
        else:
            print(f"--parallel requested but only {n_gpus} GPU visible -- ignoring, "
                  f"training on a single GPU as normal.")

    if args.compile:
        model = torch.compile(model)

    diffusion = GaussianDiffusion(model, T=args.T, beta_1=args.beta_1,
                                   beta_T=args.beta_T, device=device)
    ema_diffusion = GaussianDiffusion(ema_model, T=args.T, beta_1=args.beta_1,
                                       beta_T=args.beta_T, device=device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    def warmup_lr(step):
        if args.warmup <= 0:
            return 1.0
        return min(step, args.warmup) / args.warmup

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)

    def unwrap(m):
        """Returns the underlying module if wrapped in DataParallel, else m
        itself -- needed everywhere we touch named_parameters()/state_dict()
        directly, since DataParallel prefixes those with 'module.' and would
        silently mismatch ema_model's (always unwrapped) parameter names."""
        return m.module if isinstance(m, torch.nn.DataParallel) else m

    start_step = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        unwrap(model).load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema_model"])
        start_step = ckpt["step"]
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        else:
            print(f"WARNING: '{args.resume}' has no saved optimizer state (this is "
                  f"expected for 'latest.pt', which is saved as a lightweight copy "
                  f"without optimizer state -- see the checkpoint-saving code below). "
                  f"Resuming with a freshly-initialized optimizer (Adam momentum reset) "
                  f"instead of the exact optimizer state at this step. Minor, not a "
                  f"correctness issue, but noting it explicitly.")
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        else:
            print(f"WARNING: '{args.resume}' has no saved scheduler state -- "
                  f"fast-forwarding warmup schedule to match the resumed step count "
                  f"({start_step}) so warmup doesn't incorrectly restart mid-training.")
            for _ in range(start_step):
                scheduler.step()
        print(f"resumed from {args.resume} at step {start_step}")

    model.train()
    t0 = time.time()
    running_loss = 0.0
    amp_dtype = torch.bfloat16
    autocast_ctx = torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp)

    if args.benchmark_only:
        print(f"=== Benchmark mode: {args.benchmark_steps} steps, "
              f"amp={args.amp}, channels_last={args.channels_last}, compile={args.compile} ===")
        for _ in range(10):
            x0, _, _ = next(data_iter)
            x0 = x0.to(device, non_blocking=True)
            if args.channels_last:
                x0 = x0.to(memory_format=torch.channels_last)
            optimizer.zero_grad()
            with autocast_ctx:
                loss = diffusion.train_losses(x0)
            loss.backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()

        t_bench0 = time.time()
        for _ in range(args.benchmark_steps):
            x0, _, _ = next(data_iter)
            x0 = x0.to(device, non_blocking=True)
            if args.channels_last:
                x0 = x0.to(memory_format=torch.channels_last)
            optimizer.zero_grad()
            with autocast_ctx:
                loss = diffusion.train_losses(x0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t_bench0

        steps_per_sec = args.benchmark_steps / elapsed
        eta_hours = args.total_steps / steps_per_sec / 3600
        print(f"\nmeasured: {steps_per_sec:.3f} steps/sec  "
              f"({elapsed:.1f}s for {args.benchmark_steps} steps, batch_size={args.batch_size})")
        print(f"projected wall-clock for --total_steps={args.total_steps}: "
              f"{eta_hours:.2f} hours")
        for budget_hours in [1, 3, 6, 12, 24]:
            reachable_steps = int(steps_per_sec * budget_hours * 3600)
            print(f"  steps reachable in {budget_hours:>3}h: {reachable_steps:,}")
        return

    for step in range(start_step, args.total_steps):
        x0, _, _ = next(data_iter)
        x0 = x0.to(device, non_blocking=True)
        if args.channels_last:
            x0 = x0.to(memory_format=torch.channels_last)

        optimizer.zero_grad()
        with autocast_ctx:
            loss = diffusion.train_losses(x0)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(grad_norm):
            print(f"step {step+1}: SKIPPING optimizer step -- non-finite gradient "
                  f"(norm={grad_norm.item()}). This is the likely cause of the loss "
                  f"spikes seen in earlier runs: clip_grad_norm_ cannot rescue a NaN/Inf "
                  f"gradient (scaling by a NaN norm is a no-op), so a corrupting update "
                  f"was silently applied before. Skipping preserves the current weights "
                  f"instead of poisoning them.")
            optimizer.zero_grad()
        else:
            optimizer.step()
        scheduler.step()
        ema_update(ema_model, unwrap(model), args.ema_decay)

        running_loss += loss.item()

        if (step + 1) % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            elapsed = time.time() - t0
            print(f"step {step+1}/{args.total_steps}  loss={avg_loss:.4f}  "
                  f"elapsed={elapsed:.0f}s  steps/s={(step+1-start_step)/elapsed:.2f}")
            running_loss = 0.0

        if (step + 1) % args.sample_every == 0:
            _save_samples(ema_diffusion, device, args.T,
                          os.path.join(args.logdir, "samples", f"step{step+1}.png"))

        if (step + 1) % args.ckpt_every == 0 or (step + 1) == args.total_steps:
            ckpt_path = os.path.join(args.logdir, "ckpts", f"step{step+1}.pt")
            torch.save({
                "model": unwrap(model).state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step + 1,
                "args": vars(args),
            }, ckpt_path)
            torch.save({
                "model": unwrap(model).state_dict(),
                "ema_model": ema_model.state_dict(),
                "step": step + 1,
                "args": vars(args),
            }, os.path.join(args.logdir, "ckpts", "latest.pt"))
            print(f"saved checkpoint: {ckpt_path}")


@torch.no_grad()
def _save_samples(diffusion, device, T, path, n=64, img_size=32):
    """Quick visual sanity check during training (full ancestral sampling).

    Predicts x0 from eps and CLIPS it to [-1,1] before computing the reverse
    mean, matching SimA's actual sampler (DDPM/diffusion.py's p_mean_variance:
    x_0 = torch.clip(x_0, -1., 1.)) -- a standard DDPM stabilization trick
    that prevents compounding errors from an imperfect x0 estimate,
    especially relevant for an undertrained or recently-recovered-from-
    instability checkpoint. The original version of this function used a
    simpler direct one-step mean formula with no clipping at all."""
    diffusion.model.eval()
    x = torch.randn(n, 3, img_size, img_size, device=device)
    for t in reversed(range(T)):
        t_batch = torch.full((n,), t, dtype=torch.long, device=device)
        eps = diffusion.model(x, t_batch)
        alpha_bar_t = diffusion.alphas_bar[t]
        sqrt_recip_alphas_bar_t = 1.0 / torch.sqrt(alpha_bar_t)
        sqrt_recipm1_alphas_bar_t = torch.sqrt(1.0 / alpha_bar_t - 1.0)
        x0_pred = sqrt_recip_alphas_bar_t * x - sqrt_recipm1_alphas_bar_t * eps
        x0_pred = torch.clip(x0_pred, -1.0, 1.0)

        alpha_bar_prev = diffusion.alphas_bar[t - 1] if t > 0 else torch.tensor(1.0, device=device)
        beta_t = diffusion.betas[t]
        posterior_mean_coef1 = torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)
        posterior_mean_coef2 = (torch.sqrt(diffusion.alphas[t]) * (1.0 - alpha_bar_prev)
                                 / (1.0 - alpha_bar_t))
        mean = posterior_mean_coef1 * x0_pred + posterior_mean_coef2 * x

        if t > 0:
            posterior_var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(posterior_var) * noise
        else:
            x = mean
    x = (x.clamp(-1, 1) + 1) / 2
    save_image(x, path, nrow=8)
    diffusion.model.train()


if __name__ == "__main__":
    main()
