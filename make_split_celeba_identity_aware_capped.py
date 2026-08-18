"""
Fixes a real side effect of the original make_split_celeba_identity_aware.py
at small n: that script fills each identity COMPLETELY before moving to the
next, so hitting a small n_member (e.g. 100) only requires ~5 identities
(~20 photos each) -- enough redundancy per person that the model can
generalize across a person's face without memorizing any single photo,
silently deflating apparent memorization signal (confirmed: n=100 identity-
aware baseline showed effect size 0.0063, essentially null).

This version keeps the core leakage guarantee (every photo of a given
identity stays entirely on one side) but ALSO caps how many photos are taken
per identity (--max_per_identity), forcing many DISTINCT people into the
same n_member budget instead of a few people repeated many times -- solving
diversity collapse without reintroducing leakage risk.

Usage:
    python make_split_celeba_identity_aware_capped.py \
        --n_member 100 --max_per_identity 2 \
        --out_path ./data/celeba_split_n100_identity_aware_capped.npz
"""
import argparse
import numpy as np
import torchvision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--n_member", type=int, default=100)
    ap.add_argument("--max_per_identity", type=int, default=2,
                     help="cap photos taken per identity -- forces diversity. "
                          "e.g. n_member=100, max_per_identity=2 -> ~50 distinct "
                          "identities instead of ~5")
    ap.add_argument("--out_path", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    ds = torchvision.datasets.CelebA(root=args.data_root, split="train",
                                       target_type="identity", download=True)
    identities = np.array([int(ds[i][1]) for i in range(len(ds))])
    print(f"Total images: {len(ds)}, unique identities: {len(np.unique(identities))}")

    unique_ids = np.unique(identities)
    rng.shuffle(unique_ids)

    member_images, heldout_images = [], []
    member_ids_used, heldout_ids_used = [], []
    target_per_split = args.n_member

    for ident in unique_ids:
        img_indices = np.where(identities == ident)[0]
        rng.shuffle(img_indices)
        capped_indices = img_indices[:args.max_per_identity]  # THE FIX: cap per identity

        if len(member_images) < target_per_split:
            member_images.extend(capped_indices.tolist())
            member_ids_used.append(ident)
        elif len(heldout_images) < target_per_split:
            heldout_images.extend(capped_indices.tolist())
            heldout_ids_used.append(ident)
        else:
            break

    member_images = np.array(sorted(member_images[:target_per_split]))
    heldout_images = np.array(sorted(heldout_images[:target_per_split]))

    member_id_set = set(identities[member_images].tolist())
    heldout_id_set = set(identities[heldout_images].tolist())
    overlap = member_id_set & heldout_id_set
    assert len(overlap) == 0, f"IDENTITY LEAKAGE STILL PRESENT: {len(overlap)} overlapping identities"

    np.savez(args.out_path, member=member_images, heldout=heldout_images)
    print(f"Saved to {args.out_path}: {len(member_images)} member images "
          f"({len(member_id_set)} identities, ~{len(member_images)/max(len(member_id_set),1):.1f} "
          f"photos/identity), {len(heldout_images)} held-out images ({len(heldout_id_set)} identities)")
    print(f"Identity overlap check: {len(overlap)} (must be 0)")


if __name__ == "__main__":
    main()