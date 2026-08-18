"""
Dedicated data loading for the Guided Diffusion / ImageNet-1k experiment --
rewritten to match SimA's ACTUAL INv2_attack.py exactly (confirmed by
reading their real source), not guesswork.

Key confirmed facts from their code:
  - STRATIFIED sampling: exactly n=3 images per class (sample_per_class),
    not a flat random draw -- n=3 * 1000 classes = 3000, matching their
    stated "3k" split size.
  - Preprocessing: plain torchvision Resize(BICUBIC) + CenterCrop -- NOT
    guided-diffusion's own training-time center_crop_arr (progressive
    box-filter). This project's earlier "fix" to center_crop_arr was
    WRONG for reproducing their numbers specifically.
  - Normalization ([-1,1] scaling) is applied at ATTACK time (lambda x: x*2-1
    passed into the attacker), not in the data transform -- functionally
    equivalent to doing it in the transform, so we keep it in the transform
    here for simplicity (same end result).
  - Labels are raw ImageFolder class indices (0-999, standard alphabetical
    folder-name sort), passed DIRECTLY as the model's class condition with
    NO remapping.
  - Held-out (ImageNetV2) is loaded via plain ImageFolder pointed at the
    already-downloaded/extracted folder, same as member -- ImageNetV2Dataset
    is only used once to trigger the download/extraction.
"""
import random
from collections import defaultdict

import torch
import torchvision
import torchvision.transforms as T


def _base_transform(image_size=256):
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Lambda(lambda x: x * 2.0 - 1.0),
    ])


def sample_per_class(dataset, n, rng):
    """EXACT reimplementation of SimA's sample_per_class: n images from
    EVERY class present in dataset.samples (ImageFolder's (path, label) list).
    Raises if any class has fewer than n images available."""
    buckets = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        buckets[label].append(idx)
    subset = []
    for lbl, bucket in buckets.items():
        if len(bucket) < n:
            raise RuntimeError(f"class {lbl} has only {len(bucket)} images (need {n})")
        rng.shuffle(bucket)
        subset.extend(bucket[:n])
    rng.shuffle(subset)
    return subset


def load_imagenet_member_heldout(imagenet1k_root, imagenetv2_root, n_per_class=3, seed=2025,
                                   image_size=256):
    """
    Returns (member_images, member_labels, heldout_images, heldout_labels) --
    stacked tensors, WITH labels retained (needed for the class-conditional
    checkpoint). Matches SimA's stratified n_per_class sampling exactly.

    imagenetv2_root: path to an ALREADY-EXTRACTED ImageNetV2 folder in
    standard ImageFolder layout (one subfolder per class) -- run
    ImageNetV2Dataset(...) once separately to trigger download/extraction if
    you don't have this yet; see module docstring.
    """
    rng = random.Random(seed)
    tx = _base_transform(image_size)

    imn1k = torchvision.datasets.ImageFolder(str(imagenet1k_root), transform=tx)
    imnv2 = torchvision.datasets.ImageFolder(str(imagenetv2_root), transform=tx)

    print(f"ImageNet-1k pool: {len(imn1k)} images, {len(imn1k.classes)} classes")
    print(f"ImageNetV2 pool: {len(imnv2)} images, {len(imnv2.classes)} classes")

    idx_imn1k = sample_per_class(imn1k, n=n_per_class, rng=rng)
    idx_imnv2 = sample_per_class(imnv2, n=n_per_class, rng=rng)

    member_images, member_labels = [], []
    for i in idx_imn1k:
        x, y = imn1k[i]
        member_images.append(x)
        member_labels.append(y)

    heldout_images, heldout_labels = [], []
    for i in idx_imnv2:
        x, y = imnv2[i]
        heldout_images.append(x)
        heldout_labels.append(y)

    print(f"  member: {len(member_images)} images ({n_per_class}/class x "
          f"{len(imn1k.classes)} classes)")
    print(f"  held-out: {len(heldout_images)} images ({n_per_class}/class x "
          f"{len(imnv2.classes)} classes)")

    return (torch.stack(member_images), torch.tensor(member_labels, dtype=torch.long),
            torch.stack(heldout_images), torch.tensor(heldout_labels, dtype=torch.long))