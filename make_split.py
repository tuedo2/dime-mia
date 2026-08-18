"""
Creates the member/held-out split for STL10-U or CelebA, matching SimA's
stated split SIZES via this project's own reproducible make_member_split() --
no pre-made SecMI/SimA split file exists for these two datasets (unlike
CIFAR-10/100), so there is nothing external to convert/match.

IMPORTANT: n_member defaults below are SimA's STATED split sizes (from their
README's DDPM table), NOT n_total // 2. For STL10-U these happen to coincide
(50k/50k, and the full unlabeled pool is 100k, so half = 50k). For CelebA they
do NOT coincide -- SimA's split is 30k/30k, but CelebA's official train split
is 162,770 images, so n_total // 2 would wrongly give ~81k/81k. Always pass
--n_member explicitly if you want anything other than SimA's stated size, and
never rely on make_member_split's own n_total // 2 default for CelebA.

Usage:
    python make_split.py --dataset stl10u --out_path ./data/stl10u_split.npz
    python make_split.py --dataset celeba --out_path ./data/celeba_split.npz
"""
import argparse

from data import make_member_split, save_split, DATASET_SIZES

# SimA's STATED split sizes (README DDPM table) -- the correct n_member for
# each dataset, NOT necessarily n_total // 2.
DEFAULT_N_MEMBER = {
    "stl10u": 10000,
    "celeba": 30000,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=["stl10u", "celeba"])
    ap.add_argument("--out_path", type=str, required=True)
    ap.add_argument("--n_member", type=int, default=None,
                     help="defaults to SimA's stated split size per dataset "
                          "(50000 for stl10u, 30000 for celeba) -- override only "
                          "if you deliberately want a different size")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    n_total = DATASET_SIZES[args.dataset]
    n_member = args.n_member if args.n_member is not None else DEFAULT_N_MEMBER[args.dataset]

    member_idx, heldout_idx = make_member_split(
        dataset=args.dataset, n_total=n_total, n_member=n_member, seed=args.seed)
    save_split(member_idx, heldout_idx, args.out_path)
    print(f"Saved split to {args.out_path}: {len(member_idx)} member / "
          f"{len(heldout_idx)} held-out (dataset={args.dataset}, n_total={n_total})")


if __name__ == "__main__":
    main()