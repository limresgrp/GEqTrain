import math
import torch
from torch import nn
from scipy.special import spherical_jn


class BesselBasis(nn.Module):

    def __init__(self, r_max, num_basis=8, accuracy=1e-3, trainable=True):
        super(BesselBasis, self).__init__()

        self.trainable = trainable
        self.num_basis = num_basis
        self.num_points = int(r_max / accuracy)
        
        r_values = torch.linspace(0., r_max, self.num_points)
        Jn_values = []
        for n in range(num_basis):
            Jn_values.append(spherical_jn(n, r_values))
        bessel_values = torch.round(torch.stack(Jn_values, dim=0).float().T, decimals=6)

        self.register_buffer("r_values", r_values)
        self.register_buffer("bessel_values", bessel_values)

        bessel_weights = (
            torch.ones(self.num_basis, dtype=torch.get_default_dtype())
        )
        if self.trainable:
            self.bessel_weights = nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Bessel Basis for input x.

        Parameters
        ----------
        x : torch.Tensor
            Input
        """
        x = torch.clip(x, max=self.r_values.max())
        idcs = torch.searchsorted(self.r_values, x)
        return torch.einsum("i,ji->ji", self.bessel_weights, self.bessel_values[idcs])


class GaussianBasis(nn.Module):

    def __init__(self, r_max, num_basis=8, accuracy=1e-3, trainable=True):
        super(GaussianBasis, self).__init__()

        self.trainable = trainable
        self.num_basis = num_basis
        self.num_points = int(r_max / accuracy)
        
        r_values = torch.linspace(0., r_max, self.num_points)

        def gaussian(x, mu, sigma):
            return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / math.sqrt(2 * math.pi * sigma ** 2)
        
        G_values = []
        step = r_max / num_basis
        for n in range(num_basis):
            mu = n * step
            sigma = step
            G_values.append(gaussian(r_values, mu=mu, sigma=sigma))
        gaussian_values = torch.round(torch.stack(G_values, dim=0).float().T, decimals=6)

        self.register_buffer("r_values", r_values)
        self.register_buffer("gaussian_values", gaussian_values)

        gaussian_weights = (
            torch.ones(self.num_basis, dtype=torch.get_default_dtype())
        )
        if self.trainable:
            self.gaussian_weights = nn.Parameter(gaussian_weights)
        else:
            self.register_buffer("gaussian_weights", gaussian_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Gaussian Basis for input x.

        Parameters
        ----------
        x : torch.Tensor
            Input
        """
        x = torch.clip(x, max=self.r_values.max())
        idcs = torch.searchsorted(self.r_values, x)
        return torch.einsum("i,ji->ji", self.gaussian_weights, self.gaussian_values[idcs])