
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

class PseudoSpectra(nn.Module):
    def __init__(self, in_channels, out_channels, degree, bandwidth, bc: BoundaryType):
        super(PseudoSpectra, self).__init__()

        if bc == BoundaryType.DIRICHLET or bc == BoundaryType.PERIODIC:
            self.x2phi = x2phi_dirichlet
            self.phi2x = phi2x_dirichlet 
        elif bc == BoundaryType.NEUMANN: 
            self.x2phi = x2phi_neumann
            self.phi2x = phi2x_neumann 
        elif bc == BoundaryType.ROBIN:
            # Assume robin values are a_l, b_l, a_r, b_r 
            S_comp_to_cheb, S_cheb_to_comp = ch.get_square_robin_transforms(degree, a_L = 1.513, a_R=1.540, b_L=-1, b_R=1)
            self.register_buffer('S_comp_to_cheb', S_comp_to_cheb)
            self.register_buffer('S_cheb_to_comp', S_cheb_to_comp)
            self.x2phi = functools.partial(ch.Wrapper, [ch.dct, lambda x: ch.cmp_robin_v(x, S_cheb_to_comp = self.S_cheb_to_comp)])
            self.phi2x = functools.partial(ch.Wrapper, [lambda x: ch.icmp_robin_v(x, S_comp_to_cheb = self.S_comp_to_cheb), ch.idct])
        else:
            self.x2phi = dctn
            self.phi2x = idctn 

        
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
            b[..., : self.degree + (self.bandwidth-1)], self.weights
        )
        u = self.phi2x(out, -1)
        return u


class OPNO1D(nn.Module):
    def __init__(self, degree, width, bc: BoundaryType, output_dim: int):
        super(OPNO1D, self).__init__()

        if bc == BoundaryType.DIRICHLET or bc == BoundaryType.PERIODIC:
            self.x2phi = x2phi_dirichlet
            self.phi2x = phi2x_dirichlet 
        elif bc == BoundaryType.NEUMANN: 
            self.x2phi = x2phi_neumann
            self.phi2x = phi2x_neumann 
        elif bc == BoundaryType.ROBIN:
            # Assume robin values are a_l, b_l, a_r, b_r 
            S_comp_to_cheb, S_cheb_to_comp = ch.get_square_robin_transforms(degree, a_L = 1.513, a_R=1.540, b_L=-1, b_R=1)
            self.register_buffer('S_comp_to_cheb', S_comp_to_cheb)
            self.register_buffer('S_cheb_to_comp', S_cheb_to_comp)
            self.x2phi = functools.partial(ch.Wrapper, [ch.dct, lambda x: ch.cmp_robin_v(x, S_cheb_to_comp = self.S_cheb_to_comp)])
            self.phi2x = functools.partial(ch.Wrapper, [lambda x: ch.icmp_robin_v(x, S_comp_to_cheb = self.S_comp_to_cheb), ch.idct])
        else:
            self.x2phi = dctn
            self.phi2x = idctn 


        self.degree = degree
        self.width = width

        self.conv0 = PseudoSpectra(self.width, self.width, self.degree, 3, bc)
        self.conv1 = PseudoSpectra(self.width, self.width, self.degree, 3, bc)
        self.conv2 = PseudoSpectra(self.width, self.width, self.degree, 3, bc)
        self.conv3 = PseudoSpectra(self.width, self.width, self.degree, 3, bc)

        self.convl = PseudoSpectra(2, self.width - 2, self.degree, 3, bc)

        self.w0 = nn.Conv1d(
            self.width,
            self.width,
            1,
        )  # better
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def acti(self, x):
        return torch.nn.functional.gelu(x)

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
        # x = self.phi2x(self.x2phi(x, -2), -2)

        return x



class LayeredOPNO1D(nn.Module):
    def __init__(self, degree, width, bc: BoundaryType, nlayers: int, bandwidth: int, output_dim: int):
        super(LayeredOPNO1D, self).__init__()

        if bc == BoundaryType.DIRICHLET or bc == BoundaryType.PERIODIC:
            self.x2phi = x2phi_dirichlet
            self.phi2x = phi2x_dirichlet 
        elif bc == BoundaryType.NEUMANN: 
            self.x2phi = x2phi_neumann
            self.phi2x = phi2x_neumann 
        elif bc == BoundaryType.ROBIN:
            # Assume robin values are a_l, b_l, a_r, b_r 
            S_comp_to_cheb, S_cheb_to_comp = ch.get_square_robin_transforms(degree, a_L = 1.513, a_R=1.540, b_L=-1, b_R=1)
            self.register_buffer('S_comp_to_cheb', S_comp_to_cheb)
            self.register_buffer('S_cheb_to_comp', S_cheb_to_comp)
            self.x2phi = functools.partial(ch.Wrapper, [ch.dct, lambda x: ch.cmp_robin_v(x, S_cheb_to_comp = self.S_cheb_to_comp)])
            self.phi2x = functools.partial(ch.Wrapper, [lambda x: ch.icmp_robin_v(x, S_comp_to_cheb = self.S_comp_to_cheb), ch.idct])
        else:
            self.x2phi = dctn
            self.phi2x = idctn 


        self.degree = degree
        self.width = width

        self.convs = nn.ModuleList()
        self.ws = nn.ModuleList()

        self.fc0 = nn.Linear(2, self.width)

        for i in range(nlayers):
            self.convs.append(PseudoSpectra(self.width, self.width, self.degree, bandwidth, bc))
            self.ws.append(nn.Conv1d(self.width, self.width, 1))

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def acti(self, x):
        return torch.nn.functional.gelu(x)

    def forward(self, x):

        x = self.fc0(x)

        x = x.permute(0, 2, 1)

        for w, conv in zip(self.ws, self.convs):

            x = x + self.acti(w(x) + conv(x))

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.acti(x)
        x = self.fc2(x)

        return x

class PseudoSpectra2d(nn.Module):
    def __init__(self, in_channels, out_channels, degree1, degree2, bandwidth, bc: BoundaryType):
        super(PseudoSpectra2d, self).__init__()

        if bc == BoundaryType.DIRICHLET or bc == BoundaryType.PERIODIC:
            self.x2phi = x2phi_dirichlet
            self.phi2x = phi2x_dirichlet 
        elif bc == BoundaryType.NEUMANN: 
            self.x2phi = x2phi_neumann
            self.phi2x = phi2x_neumann 
        elif bc == BoundaryType.ROBIN:
            # Assume robin values are a_l, b_l, a_r, b_r 
            raise NotImplementedError
        else:
            self.x2phi = dctn
            self.phi2x = idctn 

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.degree1 = degree1
        self.degree2 = degree2
        self.bandwidth = bandwidth

        self.scale = 2 / (in_channels + out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels*bandwidth*bandwidth, out_channels, degree1*degree2, dtype=torch.float32))

        self.unfold = torch.nn.Unfold(kernel_size=(self.bandwidth,self.bandwidth))

    def quasi_diag_mul2d(self, input, weights):
        xpad = self.unfold(input)
        return torch.einsum("bix, iox->box", xpad, weights)
        # return torch.einsum("bixw, xiow->box", xpad, weights)

    def forward(self, u):
        batch_size, width, Nx, Ny = u.shape

        a = dctn(u, [-1, -2])

        b = torch.zeros(batch_size, self.out_channels, Nx, Ny, device=u.device, dtype=torch.float32)
        b[..., :self.degree1, :self.degree2] = \
            self.quasi_diag_mul2d(a[..., :self.degree1+2, :self.degree2+2], self.weights).reshape(
                batch_size, self.out_channels, self.degree1, self.degree2)

        u = self.phi2x(b, [-1, -2])
        return u


class OPNO2d(nn.Module):
    def __init__(self, degree1, degree2, width):
        super(OPNO2d, self).__init__()
        
        self.degree1 = degree1
        self.degree2 = degree2
        self.width = width

        self.conv0 = PseudoSpectra2d(self.width, self.width, self.degree1, self.degree2, 3)
        self.conv1 = PseudoSpectra2d(self.width, self.width, self.degree1, self.degree2, 3)
        self.conv2 = PseudoSpectra2d(self.width, self.width, self.degree1, self.degree2, 3)
        self.conv3 = PseudoSpectra2d(self.width, self.width, self.degree1, self.degree2, 3)

        self.convl = PseudoSpectra2d(3, self.width-3, self.degree1, self.degree2, 3)

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
        x = x.permute(0, 3, 1, 2)

        x = torch.cat([x, self.acti(self.convl(x))], dim=1)

        x = x+self.acti(self.w0(x) + self.conv0(x))

        x = x+self.acti(self.w1(x) + self.conv1(x))

        x = x+self.acti(self.w2(x) + self.conv2(x))

        x = x+self.acti(self.w3(x) + self.conv3(x))

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.acti(x)
        x = self.fc2(x)
        x = phi2x_dirichlet(x2phi_dirichlet(x, [1, 2]), [1, 2])
        return x


class LayeredOPNO2d(nn.Module):
    def __init__(self, degree1, degree2, width, bc: BoundaryType, nlayers: int, bandwidth: int, output_dim: int):
        super(LayeredOPNO2d, self).__init__()

        if bc == BoundaryType.DIRICHLET or bc == BoundaryType.PERIODIC:
            self.x2phi = x2phi_dirichlet
            self.phi2x = phi2x_dirichlet 
        elif bc == BoundaryType.NEUMANN: 
            self.x2phi = x2phi_neumann
            self.phi2x = phi2x_neumann 
        elif bc == BoundaryType.ROBIN:
            # Assume robin values are a_l, b_l, a_r, b_r 
            raise NotImplementedError
        else:
            self.x2phi = dctn
            self.phi2x = idctn 

        self.degree1 = degree1
        self.degree2 = degree2
        self.width = width

        self.fc0 = nn.Linear(3, self.width)

        self.convs = nn.ModuleList()
        self.ws = nn.ModuleList()

        for i in range(nlayers):
            self.convs.append(PseudoSpectra2d(self.width, self.width, self.degree1, self.degree2, bandwidth=bandwidth, bc=bc))
            self.ws.append(nn.Conv2d(self.width, self.width, 1))

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def acti(self, x):
        return torch.nn.functional.gelu(x)

    def forward(self, x):
        # x : (batches, nx, ny, [Einc(x, y), cnt(x, y), x, y])
        x = self.fc0(x)

        x = x.permute(0, 3, 1, 2)

        for conv, w in zip(self.convs, self.ws):
            x = x+self.acti(w(x) + conv(x))

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.acti(x)
        x = self.fc2(x)
        x = self.phi2x(self.x2phi(x, [1, 2]), [1, 2])
        return x