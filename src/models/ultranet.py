
import torch
import torch.nn as nn
from . import chebypack as ch
import functools

x2phi = functools.partial(ch.Wrapper, [ch.dct, ch.cmp_neumann])
phi2x = functools.partial(ch.Wrapper, [ch.icmp_neumann, ch.idct])
idctn = functools.partial(ch.Wrapper, [ch.idct])
dctn = functools.partial(ch.Wrapper, [ch.dct])


class LowRankTriangular(nn.Module):
    def __init__(self, in_channels, out_channels, v_modes, rank, is_lower=True):
        super().__init__()
        self.n = v_modes
        self.r = rank
        if is_lower:
            self.tri_func = torch.tril
        else:
            self.tri_func = torch.triu

        self.scale = 2 / (in_channels + out_channels)
        # Initialize U and V (n x r)
        # We use a standard deviation scaled by rank for stability
        self.U = nn.Parameter(
            torch.randn(out_channels, in_channels, v_modes, rank) * self.scale
        )
        self.V = nn.Parameter(
            torch.randn(out_channels, in_channels, v_modes, rank) * self.scale
        )

    def forward(self, x):
        batch_size, channels, v_modes = x.shape
        # 1. Compute the low-rank product: (n x r) @ (r x n) -> (n x n)
        matrix = torch.einsum("oivk, oiwk->oivw", self.U, self.V)
        output = torch.einsum(
            "oivw, biw->bov", self.tri_func(matrix, diagonal=-1), x
        )  # (B, out_channel, v_mode)
        return output


class BPSPseudoSpectra(nn.Module):
    def __init__(self, in_channels, out_channels, v_modes, bandwidth, rank):
        super(BPSPseudoSpectra, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.degree = v_modes
        self.bandwidth = bandwidth
        self.rank = rank

        self.scale = 2 / (in_channels + out_channels)
        self.weights = nn.Parameter(
            self.scale
            * torch.rand(
                v_modes, in_channels, out_channels, bandwidth, dtype=torch.float32
            )
        )

        self.tril = LowRankTriangular(
            in_channels, out_channels, v_modes, rank, is_lower=True
        )
        self.triu = LowRankTriangular(
            in_channels, out_channels, v_modes, rank, is_lower=False
        )

    def quasi_diag(self, x, weights):
        xpad = x.unfold(
            -1, self.bandwidth, 1
        )  # (batches, in_channel, v_mode, bandwidth)
        return torch.einsum("bixw, xiow->box", xpad, weights)

    def forward(self, u):
        # x : (batches, nx, features)
        batch_size, channels, Nx = u.shape

        b = dctn(u, -1)

        out = torch.zeros(
            batch_size, self.out_channels, Nx, device=u.device, dtype=torch.float32
        )
        out[..., : self.degree] = self.quasi_diag(
            b[..., : self.degree + 2], self.weights
        )
        L_contrib = self.tril(b[..., : self.degree])
        U_contrib = self.triu(b[..., : self.degree])

        out[..., : self.degree] += L_contrib + U_contrib

        u = phi2x(out, -1)

        return u 


class UltraNet(nn.Module):
    def __init__(self, modes, width, rank):
        super(UltraNet, self).__init__()
        self.degree = modes
        self.width = width
        self.rank = rank

        self.conv0 = BPSPseudoSpectra(self.width, self.width, self.degree, 3, self.rank)
        self.conv1 = BPSPseudoSpectra(self.width, self.width, self.degree, 3, self.rank)
        self.conv2 = BPSPseudoSpectra(self.width, self.width, self.degree, 3, self.rank)
        self.conv3 = BPSPseudoSpectra(self.width, self.width, self.degree, 3, self.rank)

        self.convl = BPSPseudoSpectra(2, self.width - 2, self.degree, 3, self.rank)

        self.w0 = nn.Conv1d(
            self.width,
            self.width,
            1,
        )  # better
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def acti(self, x):
        return nn.functional.gelu(x)

    def forward(self, x):

        x = x.permute(0, 2, 1)

        x = torch.cat([x, self.acti(self.convl(x))], dim=1)

        x = x + self.acti(self.w0(x) + self.conv0(x))

        x = x + self.acti(self.w1(x) + self.conv1(x))

        x = x + self.acti(self.w2(x) + self.conv2(x))

        x = x + self.acti(self.w3(x) + self.conv3(x))

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.acti(x)
        x = self.fc2(x)
        x = phi2x(x2phi(x, -2), -2)

        return x
