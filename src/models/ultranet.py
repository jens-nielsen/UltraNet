
import torch
import torch.nn as nn
from . import chebypack as ch
import functools
from src.utils import BoundaryType

x2phi_neumann = functools.partial(ch.Wrapper, [ch.dct, ch.cmp_neumann])
phi2x_neumann = functools.partial(ch.Wrapper, [ch.icmp_neumann, ch.idct])
x2phi_dirichlet = functools.partial(ch.Wrapper, [ch.dct, ch.cmp])
phi2x_dirichlet = functools.partial(ch.Wrapper, [ch.icmp, ch.idct])
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
    def __init__(self, in_channels, out_channels, v_modes, bandwidth, rank, bc: BoundaryType):
        super(BPSPseudoSpectra, self).__init__()

        self.phi2x = phi2x_dirichlet if bc == BoundaryType.DIRICHLET or bc == BoundaryType.PERIODIC else phi2x_neumann if bc == BoundaryType.NEUMANN else idctn 


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

        u = self.phi2x(out, -1)

        return u 


class UltraNet1D(nn.Module):
    def __init__(self, modes, width, rank,  bc: BoundaryType):
        super(UltraNet1D, self).__init__()

        self.phi2x = phi2x_dirichlet if bc == BoundaryType.DIRICHLET or bc == BoundaryType.PERIODIC else phi2x_neumann if bc == BoundaryType.NEUMANN else idctn 
        self.x2phi = x2phi_dirichlet if bc == BoundaryType.DIRICHLET or bc == BoundaryType.PERIODIC else x2phi_neumann if bc == BoundaryType.NEUMANN else dctn 

        self.degree = modes
        self.width = width
        self.rank = rank

        self.conv0 = BPSPseudoSpectra(self.width, self.width, self.degree, 3, self.rank, bc)
        self.conv1 = BPSPseudoSpectra(self.width, self.width, self.degree, 3, self.rank, bc)
        self.conv2 = BPSPseudoSpectra(self.width, self.width, self.degree, 3, self.rank, bc)
        self.conv3 = BPSPseudoSpectra(self.width, self.width, self.degree, 3, self.rank, bc)

        self.convl = BPSPseudoSpectra(2, self.width - 2, self.degree, 3, self.rank, bc)

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
        x = phi2x_neumann(x2phi_neumann(x, -2), -2)

        return x

# 2D UltraNet

class BPSPseudoSpectra2d(nn.Module):
    def __init__(self, in_channels, out_channels, degree1, degree2, bandwidth, rank):
        super(BPSPseudoSpectra2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.degree1 = degree1
        self.degree2 = degree2
        self.bandwidth = bandwidth
        self.rank = rank

        self.scale = 2 / (in_channels + out_channels)
        self.weights = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels * bandwidth * bandwidth,
                out_channels,
                degree1 * degree2,
                dtype=torch.float32,
            )
        )

        self.tril = LowRankTriangular(
            in_channels, out_channels, degree1 * degree2, rank, is_lower=True
        )
        self.triu = LowRankTriangular(
            in_channels, out_channels, degree1 * degree2, rank, is_lower=False
        )

        # self.unfold = torch.nn.Unfold(kernel_size=(self.bandwidth,self.bandwidth), padding=(self.bandwidth-1)//2)
        self.unfold = torch.nn.Unfold(kernel_size=(self.bandwidth, self.bandwidth))

    def quasi_diag_mul2d(self, input, weights):
        xpad = self.unfold(input)
        return torch.einsum("bix, iox->box", xpad, weights)

    def forward(self, u):
        batch_size, width, Nx, Ny = u.shape

        a = dctn(u, [-1, -2])

        b = torch.zeros(
            batch_size, self.out_channels, Nx, Ny, device=u.device, dtype=torch.float32
        )
        b[..., : self.degree1, : self.degree2] = self.quasi_diag_mul2d(
            a[..., : self.degree1 + 2, : self.degree2 + 2], self.weights
        ).reshape(batch_size, self.out_channels, self.degree1, self.degree2)

        # Triangular contributions - we apply the low-rank factors to the flattened coefficients and then reshape back to 2D
        L_tri = self.tril(a[..., : self.degree1, : self.degree2].flatten(-2, -1))
        U_tri = self.triu(a[..., : self.degree1, : self.degree2].flatten(-2, -1))

        b[..., : self.degree1, : self.degree2] += (L_tri + U_tri).reshape(
            batch_size, self.out_channels, self.degree1, self.degree2
        )

        u = phi2x_dirichlet(b, [-1, -2])
        return u


class UltraNet2D(nn.Module):
    def __init__(self, degree1, degree2, width, rank):
        super(UltraNet2D, self).__init__()

        self.degree1 = degree1
        self.degree2 = degree2
        self.width = width
        self.rank = rank

        self.fc0 = nn.Linear(3, self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv0 = BPSPseudoSpectra2d(self.width, self.width, self.degree1, self.degree2, 3, self.rank)
        self.conv1 = BPSPseudoSpectra2d(self.width, self.width, self.degree1, self.degree2, 3, self.rank)
        self.conv2 = BPSPseudoSpectra2d(self.width, self.width, self.degree1, self.degree2, 3, self.rank)
        self.conv3 = BPSPseudoSpectra2d(self.width, self.width, self.degree1, self.degree2, 3, self.rank)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def acti(self, x):
        return torch.nn.functional.gelu(x)

    def forward(self, x):
        # x : (batches, nx, ny, [Einc(x, y), cnt(x, y), x, y])

        x = self.fc0(x)

        x = x.permute(0, 3, 1, 2)

        x = x + self.acti(self.w0(x) + self.conv0(x))

        x = x + self.acti(self.w1(x) + self.conv1(x))

        x = x + self.acti(self.w2(x) + self.conv2(x))

        x = x + self.acti(self.w3(x) + self.conv3(x))

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.acti(x)
        x = self.fc2(x)
        x = phi2x_dirichlet(x2phi_dirichlet(x, [1, 2]), [1, 2])

        return x