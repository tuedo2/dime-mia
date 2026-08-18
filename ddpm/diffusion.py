"""
Variance-preserving forward process (Ho et al. 2020 DDPM), matching the paper's
notation throughout this project:
    beta_t: variance schedule (linear, 1e-4 -> 0.02, T=1000 -- standard CIFAR-10 config)
    alpha_t = 1 - beta_t
    alpha_bar_t = prod_{s<=t} alpha_s
    sigma_t^2 = 1 - alpha_bar_t
    x_t = sqrt(alpha_bar_t) x_0 + sigma_t * eps,  eps ~ N(0, I)

The network eps_theta(x_t, t) is trained to predict eps (the "simple" DDPM loss,
Ho et al. Eq. 14): L = E[|| eps - eps_theta(x_t, t) ||^2].

This module also exposes `query_eps`, the single shared entry point every attack
(PIA, SecMI, Loss, SimA, SimA-MC, LRT) uses to get eps_theta(x, t) for arbitrary
(possibly non-integer-batched, off-schedule) x and t -- keeping every attack
consistent in exactly how the network is called.
"""
import torch
import numpy as np


def make_beta_schedule(T=1000, beta_1=1e-4, beta_T=0.02):
    """Linear beta schedule, matching Ho et al. (2020) / pytorch-ddpm defaults."""
    return torch.linspace(beta_1, beta_T, T, dtype=torch.float64)


class GaussianDiffusion:
    """
    Precomputes and holds the VP schedule; wraps a UNet `model` (predicting eps)
    to provide training losses and a uniform query interface for all attacks.
    All schedule buffers are float64 internally (for numerical stability at the
    schedule extremes) and cast to the input dtype/device on use.
    """

    def __init__(self, model, T=1000, beta_1=1e-4, beta_T=0.02, device="cuda"):
        self.model = model
        self.T = T
        self.device = device

        betas = make_beta_schedule(T, beta_1, beta_T).to(device)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_bar = alphas_bar
        self.sqrt_alphas_bar = torch.sqrt(alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - alphas_bar)

    def _extract(self, buf, t, x_shape):
        """Index a (T,) schedule buffer at integer timesteps t: (B,), broadcast
        to x_shape = (B, C, H, W)."""
        out = buf.gather(0, t)
        return out.view(-1, *([1] * (len(x_shape) - 1))).to(torch.float32)

    def q_sample(self, x0, t, noise=None):
        """Forward process: draw x_t ~ q(x_t | x0). t: (B,) long tensor."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self._extract(self.sqrt_alphas_bar, t, x0.shape)
        sqrt_omab = self._extract(self.sqrt_one_minus_alphas_bar, t, x0.shape)
        return sqrt_ab * x0 + sqrt_omab * noise, noise

    def train_losses(self, x0, t=None):
        """Standard DDPM simple loss (Ho et al. Eq. 14). If t is None, sample
        uniformly at random per training convention."""
        if t is None:
            t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        x_t, noise = self.q_sample(x0, t)
        pred = self.model(x_t, t)
        return torch.mean((pred - noise) ** 2)

    @torch.no_grad()
    def query_eps(self, x, t):
        """
        Shared query interface for ALL attacks. x: (B, C, H, W) float tensor
        (arbitrary, not necessarily drawn from q_sample -- attacks construct
        their own inputs). t: int, or (B,) long tensor of timesteps.
        Returns eps_theta(x, t): (B, C, H, W).
        """
        self.model.eval()
        if isinstance(t, int):
            t = torch.full((x.shape[0],), t, dtype=torch.long, device=x.device)
        return self.model(x, t)

    def sqrt_alpha_bar_at(self, t):
        """sqrt(alpha_bar_t) as a python float, for scalar/int t. Used by attacks
        that need the schedule constants directly (e.g. LRT's mu_1(x))."""
        return float(self.sqrt_alphas_bar[t].item())

    def sigma_at(self, t):
        """sigma_t = sqrt(1 - alpha_bar_t) as a python float."""
        return float(self.sqrt_one_minus_alphas_bar[t].item())