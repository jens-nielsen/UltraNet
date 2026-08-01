
import torch
import torch.nn as nn
import chebypack as ch
import functools

x2phi = functools.partial(ch.Wrapper, [ch.dct, ch.cmp_neumann])
phi2x = functools.partial(ch.Wrapper, [ch.icmp_neumann, ch.idct])
idctn = functools.partial(ch.Wrapper, [ch.idct])
dctn = functools.partial(ch.Wrapper, [ch.dct])


class PseudoSpectra(nn.Module):
    def __init__(self, in_channels, out_channels, degree, bandwidth):
        super(PseudoSpectra, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.degree = degree
        self.bandwidth = bandwidth

        self.scale = 2 / (in_channels + out_channels)
        self.weights = nn.Parameter(
            self.scale
            * torch.rand(
                degree, in_channels, out_channels, bandwidth, dtype=torch.float32
            )
        )

    def quasi_diag(self, x, weights):
        xpad = x.unfold(-1, self.bandwidth, 1)
        return torch.einsum("bixw, xiow->box", xpad, weights)

    def forward(self, u):
        # x : (batches, nx, features)
        batch_size, width, Nx = u.shape

        b = dctn(u, -1)

        out = torch.zeros(
            batch_size, self.out_channels, Nx, device=u.device, dtype=torch.float32
        )
        out[..., : self.degree] = self.quasi_diag(
            b[..., : self.degree + 2], self.weights
        )

        u = phi2x(out, -1)
        return u




class OPNO(nn.Module):
    def __init__(self, modes, width):
        super(OPNO, self).__init__()
        self.degree = modes
        self.width = width

        self.conv0 = PseudoSpectra(self.width, self.width, self.degree, 3)
        self.conv1 = PseudoSpectra(self.width, self.width, self.degree, 3)
        self.conv2 = PseudoSpectra(self.width, self.width, self.degree, 3)
        self.conv3 = PseudoSpectra(self.width, self.width, self.degree, 3)
        self.conv4 = PseudoSpectra(self.width, self.width, self.degree, 3)

        self.convl = PseudoSpectra(2, self.width - 2, self.degree, 3)

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
