import math
import torch
from torch import nn
from scipy.special import spherical_jn

class BesselBasis(nn.Module):
    r_max: float
    prefactor: float

    def __init__(self, r_max, num_basis=8, trainable=True):
        r"""Radial Bessel Basis, as proposed in DimeNet: https://arxiv.org/abs/2003.03123


        Parameters
        ----------
        r_max : float
            Cutoff radius

        num_basis : int
            Number of Bessel Basis functions

        trainable : bool
            Train the :math:`n \pi` part or not.
        """
        super(BesselBasis, self).__init__()

        self.trainable = trainable
        self.num_basis = num_basis

        self.r_max = float(r_max)
        self.prefactor = 2.0 / self.r_max

        bessel_weights = (
            torch.linspace(start=1.0, end=num_basis, steps=num_basis) * math.pi
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
        numerator = torch.sin(self.bessel_weights * x.unsqueeze(-1) / self.r_max)

        return self.prefactor * (numerator / x.unsqueeze(-1))


class BesselBasisVec(nn.Module):

    def __init__(self, r_max, num_basis=8, accuracy=1e-3, trainable=True):
        super(BesselBasisVec, self).__init__()

        self.trainable = trainable
        self.num_basis = num_basis
        self.num_points = int(r_max / accuracy)

        r_values = torch.linspace(0., r_max, self.num_points)
        Jn_values = []
        for n in range(num_basis):
            Jn_values.append(spherical_jn(n, r_values))
        bessel_values = torch.stack(Jn_values, dim=0).float().T

        self.register_buffer("r_values", r_values)
        self.register_buffer("bessel_values", bessel_values)

        bessel_weights = (
            torch.ones(self.num_basis, dtype=torch.float32)
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
        gaussian_values = torch.stack(G_values, dim=0).float().T

        self.register_buffer("r_values", r_values)
        self.register_buffer("gaussian_values", gaussian_values)

        gaussian_weights = (
            torch.ones(self.num_basis, dtype=torch.float32)
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


class PolyBasisVec(nn.Module):

    def __init__(self, r_max, num_basis=8, accuracy=1e-3, trainable=True):
        super(PolyBasisVec, self).__init__()

        self.trainable = trainable
        self.num_basis = num_basis
        self.num_points = int(r_max / accuracy)

        r_values = torch.linspace(0., r_max, self.num_points)

        def poly(x, power):
            return torch.pow(x, -power)

        P_values = []
        for power in range(1, num_basis + 1):
            P_values.append(poly(r_values, power=power))
        poly_values = torch.stack(P_values, dim=0).float().T

        self.register_buffer("r_values", r_values)
        self.register_buffer("poly_values", poly_values)

        poly_weights = (
            torch.ones(self.num_basis, dtype=torch.float32)
        )
        if self.trainable:
            self.poly_weights = nn.Parameter(poly_weights)
        else:
            self.register_buffer("poly_weights", poly_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Poly Basis for input x.

        Parameters
        ----------
        x : torch.Tensor
            Input
        """
        x = torch.clip(x, max=self.r_values.max())
        idcs = torch.searchsorted(self.r_values, x)
        return torch.einsum("i,ji->ji", self.poly_weights, self.poly_values[idcs])