"""
Dataset loading and member/held-out split -- CIFAR-10/100/STL10-U/CelebA.

CIFAR-10/100 share torchvision's (root, train, download, transform) API and
32x32 native resolution -- a direct parameterization (see DATASET_SIZES /
_load_cifar_style below). STL10-U and CelebA do NOT share this API, and are
NOT 32x32 natively -- each gets its own loader function (_load_stl10u,
_load_celeba) that (a) uses the correct torchvision constructor signature for
that dataset, and (b) resizes/crops to 32x32 to match this project's DDPM
architecture (ch=128, ch_mult=(1,2,2,2), attn at 16x16 -- all sized for 32x32
input).

Images are scaled to [-1, 1] (standard DDPM convention, matching Ho et al. 2020).
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as T


# --- CIFAR-style datasets: share the (root, train, download, transform) API ---
CIFAR_STYLE_REGISTRY = {
    "cifar10": torchvision.datasets.CIFAR10,
    "cifar100": torchvision.datasets.CIFAR100,
}

# n_total = full pool size used for member+heldout selection (make_member_split
# draws member/heldout as two DISJOINT halves of this pool -- for STL10-U this
# pool is the 100k unlabeled set; for CelebA it's the ~162,770-image official
# train split, of which we only use 60k, well within its size).
DATASET_SIZES = {
    "cifar10": 50000,
    "cifar100": 50000,
    "stl10u": 100000,   # full unlabeled pool; 10k/10k drawn from it
    "celeba": 162770,   # official CelebA train split size; 30k/30k drawn from it
}


def _cifar_style_loader(name, root, transform, split):
    cls = CIFAR_STYLE_REGISTRY[name]
    return cls(root=root, train=(split == "train"), download=True, transform=transform)


def _stl10u_loader(root, transform, split):
    """
    STL10-U: torchvision.datasets.STL10 with split='unlabeled' for the 100k
    member/held-out pool. For the SHADOW set (auxiliary data never touched by
    member/held-out selection), use STL10's separate LABELED 'test' split
    (8,000 images) -- genuinely disjoint from the unlabeled pool by
    construction (torchvision keeps labeled/unlabeled splits as different
    files), matching the "shadow set = official test split" pattern used for
    CIFAR-10/100.
    """
    if split == "train":
        return torchvision.datasets.STL10(root=root, split="unlabeled",
                                           download=True, transform=transform)
    elif split == "shadow":
        return torchvision.datasets.STL10(root=root, split="test",
                                           download=True, transform=transform)
    else:
        raise ValueError(f"stl10u: unknown split '{split}'")


def _celeba_loader(root, transform, split):
    """
    CelebA: torchvision.datasets.CelebA with its own split names
    ('train'/'valid'/'test'). Member/held-out pool drawn from the official
    'train' split (162,770 images); shadow set uses the official 'test' split
    (19,962 images), genuinely disjoint by construction.

    NOTE: torchvision's CelebA download can be rate-limited/unreliable via the
    default Google Drive source -- if download=True fails repeatedly, download
    the aligned&cropped images manually and place them under
    <root>/celeba/ following torchvision's expected layout, then this loader
    will find them without re-downloading.
    """
    if split == "train":
        return torchvision.datasets.CelebA(root=root, split="train",
                                            download=True, transform=transform)
    elif split == "shadow":
        return torchvision.datasets.CelebA(root=root, split="test",
                                            download=True, transform=transform)
    else:
        raise ValueError(f"celeba: unknown split '{split}'")


# Native resolution / crop handling per dataset, applied BEFORE the standard
# [-1,1] scaling. CIFAR-10/100 are already 32x32, no resize needed.
def _resize_transform_for(name):
    if name in CIFAR_STYLE_REGISTRY:
        return []  # already 32x32
    if name == "stl10u":
        # STL10 images are 96x96 -- direct resize to 32x32 (no crop; STL10-U
        # images are already roughly object-centered, unlike CelebA's faces
        # with variable framing).
        return [T.Resize((32, 32))]
    if name == "celeba":
        # CelebA images are ~178x218, off-center faces with background --
        # standard DDPM CelebA preprocessing (Ho et al.-style): center crop to
        # a square region tightly around the face, then resize to target res.
        # 140x140 center crop is a common choice in this literature; adjust if
        # you have a reason to prefer a different crop size.
        return [T.CenterCrop(140), T.Resize((32, 32))]
    raise ValueError(f"No resize transform defined for dataset '{name}'")


def _get_loader_fn(name):
    name = name.lower()
    if name in CIFAR_STYLE_REGISTRY:
        return lambda root, transform, split: _cifar_style_loader(name, root, transform, split)
    if name == "stl10u":
        return _stl10u_loader
    if name == "celeba":
        return _celeba_loader
    raise ValueError(
        f"Unknown dataset '{name}'. Registered: "
        f"{list(CIFAR_STYLE_REGISTRY.keys()) + ['stl10u', 'celeba']}. "
        f"If adding a new one, verify its torchvision API and native "
        f"resolution first -- do not assume it matches CIFAR's.")


def get_train_split(root, dataset="cifar10"):
    """Full train/member-pool split, scaled to [-1, 1], with random-flip
    augmentation (standard DDPM training augmentation)."""
    loader_fn = _get_loader_fn(dataset)
    transform = T.Compose(_resize_transform_for(dataset.lower()) + [
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Lambda(lambda x: x * 2.0 - 1.0),
    ])
    return loader_fn(root, transform, "train")


def get_train_split_noaug(root, dataset="cifar10"):
    """Same, but without random flip -- used for attack evaluation."""
    loader_fn = _get_loader_fn(dataset)
    transform = T.Compose(_resize_transform_for(dataset.lower()) + [
        T.ToTensor(),
        T.Lambda(lambda x: x * 2.0 - 1.0),
    ])
    return loader_fn(root, transform, "train")


def get_shadow_split(root, dataset="cifar10"):
    """Official test/shadow split, never touched by member/held-out selection."""
    loader_fn = _get_loader_fn(dataset)
    transform = T.Compose(_resize_transform_for(dataset.lower()) + [
        T.ToTensor(),
        T.Lambda(lambda x: x * 2.0 - 1.0),
    ])
    dataset_l = dataset.lower()
    split_name = "shadow" if dataset_l in ("stl10u", "celeba") else "test"
    if split_name == "test":
        return loader_fn(root, transform, "test")
    return loader_fn(root, transform, "shadow")


def make_member_split(dataset="cifar10", n_total=None, n_member=None, seed=0):
    """
    Reproducible half/half partition of the dataset's train/member-pool
    indices. Returns (member_indices, heldout_indices), both sorted int arrays.
    """
    if n_total is None:
        n_total = DATASET_SIZES[dataset.lower()]
    if n_member is None:
        n_member = n_total // 2
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    member_idx = np.sort(perm[:n_member])
    heldout_idx = np.sort(perm[n_member:n_member + n_member])
    return member_idx, heldout_idx


def save_split(member_idx, heldout_idx, path):
    np.savez(path, member=member_idx, heldout=heldout_idx)


def load_split(path):
    d = np.load(path)
    return d["member"], d["heldout"]


def convert_secmi_split(secmi_npz_path, out_path):
    """
    Converts SecMI's own split file format into our save_split format. Only
    applicable to CIFAR-10/100, which have real SecMI-released split files --
    no equivalent exists for STL10-U/CelebA (see module docstring); use
    make_member_split() + save_split() directly for those instead.
    """
    d = np.load(secmi_npz_path)
    member_idx = d["mia_train_idxs"]
    heldout_idx = d["mia_eval_idxs"]
    assert len(np.intersect1d(member_idx, heldout_idx)) == 0, \
        "SecMI split indices overlap -- unexpected, do not proceed without investigating"
    save_split(member_idx, heldout_idx, out_path)
    print(f"Converted {secmi_npz_path} -> {out_path} "
          f"({len(member_idx)} member / {len(heldout_idx)} held-out)")
    return member_idx, heldout_idx


class IndexedSubset(Dataset):
    """Like torch.utils.data.Subset, but also returns the global index."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        x, y = self.dataset[idx]
        return x, y, idx


def get_member_heldout_loaders(root, split_path, dataset="cifar10", batch_size=128,
                                num_workers=4, for_training=True):
    """
    for_training=True: member loader uses random-flip augmentation (for DDPM
        training). for_training=False: no augmentation (for attack evaluation).
    Returns (member_loader, heldout_loader, member_idx, heldout_idx).
    """
    if os.path.exists(split_path):
        member_idx, heldout_idx = load_split(split_path)
    else:
        member_idx, heldout_idx = make_member_split(dataset=dataset)
        os.makedirs(os.path.dirname(split_path), exist_ok=True)
        save_split(member_idx, heldout_idx, split_path)

    base = (get_train_split(root, dataset=dataset) if for_training
            else get_train_split_noaug(root, dataset=dataset))
    member_ds = IndexedSubset(base, member_idx)
    heldout_ds = IndexedSubset(base, heldout_idx)

    member_loader = DataLoader(member_ds, batch_size=batch_size, shuffle=for_training,
                                num_workers=num_workers, drop_last=for_training,
                                pin_memory=True)
    heldout_loader = DataLoader(heldout_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)
    return member_loader, heldout_loader, member_idx, heldout_idx


# --- Backward-compat aliases ---
def get_cifar10_train(root):
    return get_train_split(root, dataset="cifar10")


def get_cifar10_train_noaug(root):
    return get_train_split_noaug(root, dataset="cifar10")
