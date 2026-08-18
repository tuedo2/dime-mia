"""
Adapter for OpenAI's guided-diffusion, CLASS-CONDITIONAL checkpoint --
rewritten after confirming via SimA's actual INv2_attack.py that their
reported numbers use 256x256_diffusion.pt (class_cond=True).

Download the correct checkpoint:
    wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt \
        -O models/256x256_diffusion.pt

Label-passing design: rather than modifying every attacks/*.py function's
signature to accept a class label (touching many files across the project),
EpsOnlyWrapper exposes a settable `current_labels` attribute. Set it once per
image batch immediately BEFORE calling any attack function on that batch;
every subsequent query_eps(x, t) call for that batch will automatically use
it. This keeps attacks/*.py completely unchanged and backward-compatible
with every other (unconditional) dataset in this project.
"""
import sys
import torch
import torch.nn as nn

GUIDED_DIFFUSION_REPO_PATH = "./guided-diffusion"
sys.path.insert(0, GUIDED_DIFFUSION_REPO_PATH)

from guided_diffusion.script_util import model_and_diffusion_defaults, create_model_and_diffusion  # noqa: E402

from ddpm.diffusion import GaussianDiffusion  # noqa: E402


# Matches SimA's get_model() config exactly (confirmed from their actual code):
#   class_cond=True, learn_sigma=True, num_channels=256, num_head_channels=64,
#   attention_resolutions="32,16,8", resblock_updown=True,
#   use_scale_shift_norm=True, use_fp16=False, timestep_respacing=""
GUIDED_DIFFUSION_COND_256_CONFIG = dict(
    image_size=256,
    num_channels=256,
    num_res_blocks=2,
    num_head_channels=64,
    attention_resolutions="32,16,8",
    dropout=0.0,
    learn_sigma=True,
    class_cond=True,          # <-- the key change from the earlier (wrong) adapter
    diffusion_steps=1000,
    noise_schedule="linear",
    resblock_updown=True,
    use_scale_shift_norm=True,
    use_fp16=False,
)


class EpsOnlyWrapper(nn.Module):
    """
    Wraps guided-diffusion's UNetModel (class-conditional, learn_sigma=True)
    to match this project's expected model(x, t) -> eps signature, WITHOUT
    requiring every call site to also pass a label explicitly.

    Usage:
        wrapped_model.current_labels = label_batch   # set once per image batch
        diffusion.query_eps(x, t)                     # uses it automatically
        wrapped_model.current_labels = None            # optional: clear after
    """

    def __init__(self, unet_model, in_channels=3):
        super().__init__()
        self.unet_model = unet_model
        self.in_channels = in_channels
        self.current_labels = None  # (B,) long tensor of class indices, or None

    def forward(self, x, t):
        if self.current_labels is not None:
            assert self.current_labels.shape[0] == x.shape[0], (
                f"current_labels batch size ({self.current_labels.shape[0]}) doesn't "
                f"match input batch size ({x.shape[0]}) -- did you forget to update "
                f"current_labels for this batch, or is a query internally re-batching "
                f"(e.g. repeat_interleave for q>1) without updating labels to match?"
            )
            out = self.unet_model(x, t, y=self.current_labels)
        else:
            out = self.unet_model(x, t, y=None)
        return out[:, : self.in_channels]


def load_guided_diffusion(checkpoint_path, device="cuda", T=1000):
    """
    Loads the 256x256 CLASS-CONDITIONAL guided-diffusion checkpoint. Returns
    a standard ddpm.diffusion.GaussianDiffusion instance whose .model is the
    EpsOnlyWrapper (with the settable .current_labels attribute) -- access it
    via `diffusion.model.current_labels = ...` before each batch.
    """
    args = model_and_diffusion_defaults()
    args.update(GUIDED_DIFFUSION_COND_256_CONFIG)

    unet_model, _gd_diffusion = create_model_and_diffusion(**args)

    state_dict = torch.load(checkpoint_path, map_location=device)
    unet_model.load_state_dict(state_dict)
    unet_model.to(device)
    unet_model.eval()
    for p in unet_model.parameters():
        p.requires_grad_(False)

    wrapped_model = EpsOnlyWrapper(unet_model, in_channels=3).to(device)

    diffusion = GaussianDiffusion(wrapped_model, T=T, beta_1=1e-4, beta_T=0.02, device=device)
    return diffusion