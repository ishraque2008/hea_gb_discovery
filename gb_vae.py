"""
gb_vae.py
---------
Variational Autoencoder for HEA grain-boundary structural descriptors.

Input:  SOAP / MBTR descriptor vectors (float32, shape [N, descriptor_dim])
Output: Reconstructed descriptors + 16-dimensional latent space

Architecture
------------
  Encoder: descriptor_dim -> 256 -> 128 -> (mu, logvar) [latent_dim]
  Decoder: latent_dim     -> 128 -> 256 -> descriptor_dim

Physics-penalized reconstruction loss
  L = MSE(x_hat, x)  +  beta * KL(q(z|x) || p(z))  +  gamma * phys_penalty(x_hat)

where phys_penalty discourages descriptor values outside a learned valid range.

Usage
-----
  from gb_vae import GBVAE, vae_loss, train_vae
  import numpy as np

  X = np.load("descriptors.npy")          # shape [N, D]
  model, history = train_vae(X, epochs=100)
  z_mu, z_logvar = model.encode(torch.tensor(X, dtype=torch.float32))
  x_new = model.decode(torch.randn(10, 16))  # sample 10 new GB configs
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


# ── Model ──────────────────────────────────────────────────────────────────

class GBVAE(nn.Module):
    """
    Variational Autoencoder for grain-boundary SOAP/MBTR descriptor vectors.

    Parameters
    ----------
    descriptor_dim : int
        Dimensionality of input descriptor (e.g. 256 for SOAP with 4 species).
    latent_dim : int
        Dimensionality of latent space (default 16).
    hidden_dims : list[int]
        Hidden layer sizes for encoder/decoder (default [256, 128]).
    """

    def __init__(self, descriptor_dim: int, latent_dim: int = 16,
                 hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.descriptor_dim = descriptor_dim
        self.latent_dim = latent_dim

        # Encoder
        enc_layers = []
        in_dim = descriptor_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.SiLU()]
            in_dim = h
        self.encoder_net = nn.Sequential(*enc_layers)
        self.fc_mu     = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

        # Decoder
        dec_layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.SiLU()]
            in_dim = h
        dec_layers += [nn.Linear(in_dim, descriptor_dim)]
        self.decoder_net = nn.Sequential(*dec_layers)

        # Learnable descriptor range for physics penalty
        self.register_buffer('desc_min', torch.zeros(descriptor_dim))
        self.register_buffer('desc_max', torch.ones(descriptor_dim))

    def encode(self, x: torch.Tensor):
        """Return (mu, logvar) for input descriptor batch x."""
        h = self.encoder_net(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector z to descriptor space."""
        return self.decoder_net(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    def set_descriptor_range(self, X: np.ndarray):
        """
        Fit the physical validity range from training data.
        Called once before training; used in physics penalty.
        """
        self.desc_min = torch.tensor(X.min(axis=0), dtype=torch.float32)
        self.desc_max = torch.tensor(X.max(axis=0), dtype=torch.float32)

    def sample(self, n: int, device: str = 'cpu') -> torch.Tensor:
        """Sample n new GB configurations from the prior p(z) = N(0, I)."""
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)


# ── Loss ───────────────────────────────────────────────────────────────────

def vae_loss(x: torch.Tensor, x_hat: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             desc_min: torch.Tensor, desc_max: torch.Tensor,
             beta: float = 1.0, gamma: float = 0.1) -> dict:
    """
    Combined VAE loss:
      recon  = mean squared error over descriptor dimensions
      kl     = KL divergence from unit Gaussian prior (mean over batch)
      phys   = physics penalty: mean ReLU violation outside [desc_min, desc_max]
      total  = recon + beta * kl + gamma * phys

    Returns dict with all four scalar loss values.
    """
    recon = nn.functional.mse_loss(x_hat, x, reduction='mean')

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Physical range violation: penalise x_hat outside training data range
    lower_viol = torch.relu(desc_min - x_hat).mean()
    upper_viol = torch.relu(x_hat - desc_max).mean()
    phys = lower_viol + upper_viol

    total = recon + beta * kl + gamma * phys
    return {'total': total, 'recon': recon, 'kl': kl, 'phys': phys}


# ── Training ───────────────────────────────────────────────────────────────

def train_vae(X: np.ndarray, latent_dim: int = 16, hidden_dims: list = None,
              epochs: int = 200, batch_size: int = 64, lr: float = 1e-3,
              beta: float = 1.0, gamma: float = 0.1,
              device: str = 'cpu', verbose: bool = True) -> tuple:
    """
    Train GBVAE on descriptor array X.

    Parameters
    ----------
    X : np.ndarray, shape [N, D]
        SOAP/MBTR descriptor matrix (will be z-score normalised internally).
    Returns
    -------
    model : GBVAE (in eval mode)
    history : dict with keys 'total', 'recon', 'kl', 'phys' (lists of epoch losses)
    scaler : (mean, std) tuple for de-normalisation
    """
    # Normalise
    mu_data  = X.mean(axis=0, keepdims=True)
    std_data = X.std(axis=0, keepdims=True) + 1e-8
    X_norm   = ((X - mu_data) / std_data).astype(np.float32)

    descriptor_dim = X_norm.shape[1]
    model = GBVAE(descriptor_dim, latent_dim=latent_dim,
                  hidden_dims=hidden_dims or [256, 128]).to(device)
    model.set_descriptor_range(X_norm)

    dataset = TensorDataset(torch.tensor(X_norm, dtype=torch.float32))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    opt     = optim.Adam(model.parameters(), lr=lr)

    history = {'total': [], 'recon': [], 'kl': [], 'phys': []}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = {k: 0.0 for k in history}
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad()
            x_hat, mu_z, logvar_z = model(batch)
            losses = vae_loss(batch, x_hat, mu_z, logvar_z,
                              model.desc_min, model.desc_max,
                              beta=beta, gamma=gamma)
            losses['total'].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k in epoch_losses:
                epoch_losses[k] += losses[k].item() * len(batch)

        n = len(X_norm)
        for k in history:
            history[k].append(epoch_losses[k] / n)

        if verbose and epoch % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch:4d}/{epochs}  "
                  f"loss={history['total'][-1]:.4f}  "
                  f"recon={history['recon'][-1]:.4f}  "
                  f"kl={history['kl'][-1]:.4f}  "
                  f"phys={history['phys'][-1]:.5f}")

    model.eval()
    scaler = (mu_data, std_data)
    return model, history, scaler


# ── Utility ────────────────────────────────────────────────────────────────

def encode_descriptors(model: GBVAE, X: np.ndarray,
                       scaler: tuple, device: str = 'cpu') -> np.ndarray:
    """
    Encode descriptor matrix X to latent mu vectors.
    Returns np.ndarray of shape [N, latent_dim].
    """
    mu_data, std_data = scaler
    X_norm = ((X - mu_data) / (std_data + 1e-8)).astype(np.float32)
    with torch.no_grad():
        x_t = torch.tensor(X_norm).to(device)
        mu_z, _ = model.encode(x_t)
    return mu_z.cpu().numpy()


def save_vae(model: GBVAE, scaler: tuple, path: str = 'gb_vae.pt'):
    torch.save({'state_dict': model.state_dict(),
                'descriptor_dim': model.descriptor_dim,
                'latent_dim': model.latent_dim,
                'scaler': scaler}, path)
    print(f"VAE saved to {path}")


def load_vae(path: str = 'gb_vae.pt', device: str = 'cpu') -> tuple:
    ckpt   = torch.load(path, map_location=device)
    model  = GBVAE(ckpt['descriptor_dim'], ckpt['latent_dim'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, ckpt['scaler']
