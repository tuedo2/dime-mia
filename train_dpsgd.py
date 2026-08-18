"""
DP-SGD training variant of train.py, for the defense-evaluation experiment.
Reuses the SAME architecture/data pipeline; only the training loop and
optimizer wrapping change (Opacus, per-sample gradient clipping + noise).

CRITICAL DESIGN PRINCIPLE: run this on the SAME dataset/split/step-budget as
a checkpoint you've already CONFIRMED is vulnerable without DP (e.g. STL10-U
v4's 10k-member split) -- otherwise "the attack fails" is confounded with
"there was nothing to attack in the first place."

REQUIRED FIRST STEP -- check architecture compatibility BEFORE a long run:
    python -c "
    from opacus.validators import ModuleValidator
    from ddpm.model_dpsgd import UNet
    model = UNet(T=1000, ch=128)
    errors = ModuleValidator.validate(model, strict=False)
    print(f'{len(errors)} compatibility issues found:')
    for e in errors: print(' ', e)
    "
If this reports issues (most likely around attention layers), ModuleValidator.fix()
below will attempt an automatic fix -- but that MAY alter the architecture (e.g.
replacing an incompatible layer), which is worth knowing about explicitly, not
silently accepting.

Usage:
    python -u train_dpsgd.py --data_root ./data --dataset celeba \
        --split_path ./data/celeba_split_n1000_identity_aware_capped.npz \
        --logdir ./logs/celeba_ddpm_dpsgd_eps1_n1000 \
        --target_epsilon 1.0 --epochs 500 --batch_size 32 --max_physical_batch_size 32 \
        --ckpt_every_epochs 100 \
        2>&1 | tee results/celeba_train/dpsgd_eps1_n1000_output.txt
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from ddpm.model_dpsgd import UNet
from ddpm.diffusion import GaussianDiffusion
from data import get_member_heldout_loaders, get_train_split_noaug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--split_path", type=str, required=True)
    ap.add_argument("--logdir", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=128,
                     help="LOGICAL batch size for privacy accounting -- actual physical "
                          "sub-batch size is controlled separately via "
                          "--max_physical_batch_size (memory management)")
    ap.add_argument("--max_physical_batch_size", type=int, default=32,
                     help="per-sample gradient computation is memory-hungry -- this caps "
                          "the actual GPU batch, with BatchMemoryManager transparently "
                          "accumulating up to --batch_size before each optimizer step. "
                          "Lower this if you OOM.")
    ap.add_argument("--epochs", type=int, default=4096,
                     help="Opacus needs epoch count upfront for privacy accounting -- "
                          "match this to give the SAME total exposure as your confirmed-"
                          "vulnerable non-DP checkpoint (e.g. v4's 4096 epochs)")
    ap.add_argument("--target_epsilon", type=float, default=10.0,
                     help="standard, oft-cited 'moderate' DP budget in the image-generation "
                          "DP-SGD literature. Lower = stronger privacy, typically worse "
                          "sample quality. Worth explicitly justifying this choice in the "
                          "paper rather than treating it as a default.")
    ap.add_argument("--target_delta", type=float, default=1e-5)
    ap.add_argument("--max_grad_norm", type=float, default=1.0,
                     help="per-sample gradient clipping threshold -- standard DP-SGD "
                          "hyperparameter, distinct from train.py's batch-level grad_clip")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ch", type=int, default=128)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--ckpt_every_epochs", type=int, default=200)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join(args.logdir, "ckpts"), exist_ok=True)

    model = UNet(T=args.T, ch=args.ch).to(device)

    # --- MANDATORY compatibility check, not optional ---
    from opacus.validators import ModuleValidator
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        print(f"WARNING: {len(errors)} DP-incompatibility issue(s) found in the architecture:")
        for e in errors:
            print(f"  {e}")
        print("Attempting automatic fix via ModuleValidator.fix() -- THIS MAY ALTER THE "
              "ARCHITECTURE (e.g. replacing an incompatible layer). Compare the fixed "
              "model's structure against the original before trusting results from a long run.")
        model = ModuleValidator.fix(model)
        remaining = ModuleValidator.validate(model, strict=False)
        if remaining:
            raise RuntimeError(f"{len(remaining)} issue(s) remain after auto-fix -- "
                                f"manual architecture changes needed before DP-SGD training "
                                f"is possible: {remaining}")
        print("Auto-fix resolved all compatibility issues.")
    else:
        print("Architecture is DP-compatible with no changes needed "
              "(expected, given GroupNorm-only normalization).")

    diffusion = GaussianDiffusion(model, T=args.T, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    _, _, member_idx, _ = get_member_heldout_loaders(
        args.data_root, args.split_path, dataset=args.dataset, batch_size=1,
        num_workers=0, for_training=False)
    base = get_train_split_noaug(args.data_root, dataset=args.dataset)  # no aug -- Opacus
    # per-sample gradient hooks can be sensitive to certain augmentation/randomness patterns;
    # kept simple and matching the noaug convention used elsewhere for eval-time consistency.
    member_dataset = torch.utils.data.Subset(base, member_idx)
    dataloader = DataLoader(member_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=4, drop_last=True)

    print(f"Member set: {len(member_dataset)} images, {args.epochs} epochs, "
          f"target_epsilon={args.target_epsilon}, target_delta={args.target_delta}")

    from opacus import PrivacyEngine
    privacy_engine = PrivacyEngine()
    model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
        module=model, optimizer=optimizer, data_loader=dataloader,
        target_epsilon=args.target_epsilon, target_delta=args.target_delta,
        epochs=args.epochs, max_grad_norm=args.max_grad_norm)
    diffusion.model = model  # swap in the Opacus-wrapped model
    print(f"Noise multiplier computed by Opacus: {optimizer.noise_multiplier:.4f}")

    from opacus.utils.batch_memory_manager import BatchMemoryManager

    step = 0
    for epoch in range(args.epochs):
        with BatchMemoryManager(data_loader=dataloader,
                                  max_physical_batch_size=args.max_physical_batch_size,
                                  optimizer=optimizer) as memory_safe_dataloader:
            for x0, _ in memory_safe_dataloader:
                x0 = x0.to(device)
                optimizer.zero_grad()
                loss = diffusion.train_losses(x0)
                loss.backward()
                optimizer.step()
                step += 1
                if step % 100 == 0:
                    eps_so_far = privacy_engine.get_epsilon(args.target_delta)
                    print(f"epoch {epoch} step {step}: loss={loss.item():.4f} "
                          f"eps_so_far={eps_so_far:.2f}")

        if (epoch + 1) % args.ckpt_every_epochs == 0 or (epoch + 1) == args.epochs:
            eps_final = privacy_engine.get_epsilon(args.target_delta)
            torch.save({
                "model": model._module.state_dict(),  # unwrap Opacus's GradSampleModule
                "epoch": epoch + 1,
                "epsilon_spent": eps_final,
                "args": vars(args),
            }, os.path.join(args.logdir, "ckpts", f"epoch{epoch+1}.pt"))
            torch.save({
                "model": model._module.state_dict(),
                "epoch": epoch + 1,
                "epsilon_spent": eps_final,
                "args": vars(args),
            }, os.path.join(args.logdir, "ckpts", "latest.pt"))
            print(f"saved checkpoint at epoch {epoch+1}, epsilon spent so far: {eps_final:.2f}")

    print(f"Done. Final epsilon spent: {privacy_engine.get_epsilon(args.target_delta):.2f} "
          f"(target was {args.target_epsilon})")


if __name__ == "__main__":
    main()