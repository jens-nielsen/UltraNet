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


class UltrasphericalInverse(nn.Module):
    def __init__(self, in_c: int, out_c: int, modes: int, num_bcs: int, semi_rank: int):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.modes = modes
        self.k = num_bcs           # Rank for Woodbury correction and boundary blocks
        self.semi_rank = semi_rank # Rank for the semiseparable (banded inverse) part
        self.interior_len = modes - num_bcs
        self.scale = 2/(in_c + out_c)

        # ---------------------------------------------------------
        # 1. The B22 Block (Interior): Semiseparable + Rank k
        # ---------------------------------------------------------
        # Semiseparable generators (Rank determined by PDE bandwidth/stencil)
        self.P = nn.Parameter(torch.randn(in_c, out_c, self.interior_len, self.semi_rank) * self.scale)
        self.Q = nn.Parameter(torch.randn(in_c, out_c, self.interior_len, self.semi_rank) * self.scale)
        self.U_semi = nn.Parameter(torch.randn(in_c, out_c, self.interior_len, self.semi_rank) * self.scale)
        self.V_semi = nn.Parameter(torch.randn(in_c, out_c, self.interior_len, self.semi_rank) * self.scale)
        self.diag = nn.Parameter(torch.randn(in_c, out_c, self.interior_len) * self.scale)
        
        # Low-Rank generators (Woodbury correction, rank determined by num_bcs)
        self.U_lr = nn.Parameter(torch.randn(in_c, out_c, self.interior_len, self.k) * self.scale)
        self.V_lr = nn.Parameter(torch.randn(in_c, out_c, self.interior_len, self.k) * self.scale)

        # ---------------------------------------------------------
        # 2. The Dense Boundary Blocks: B11, B12, B21
        # ---------------------------------------------------------
        # B11: Boundary to Boundary (num_bcs x num_bcs)
        self.B11 = nn.Parameter(torch.randn(in_c, out_c, self.k, self.k) * self.scale)
        
        # B12: Interior to Boundary (num_bcs x interior_len)
        self.B12 = nn.Parameter(torch.randn(in_c, out_c, self.k, self.interior_len) * self.scale)
        
        # B21: Boundary to Interior (interior_len x num_bcs)
        self.B21 = nn.Parameter(torch.randn(in_c, out_c, self.interior_len, self.k)* self.scale)

    def ultraspherical_forward(self, x):
        """
        x shape: (batch_size, in_c, seq_len)
        y shape: (batch_size, out_c, seq_len)
        """
        # --- Form B22 ---
        # Semiseparable part 
        # (The 'k' in the einsum string is a dummy index handling self.semi_rank)
        lower_dense = torch.einsum('iotk, ioSk -> iotS', self.P, self.Q)
        upper_dense = torch.einsum('iotk, ioSk -> iotS', self.U_semi, self.V_semi)
        B22_semi = torch.tril(lower_dense) + torch.triu(upper_dense, diagonal=1)
        
        # Add diagonal
        eye = torch.eye(self.interior_len, device=x.device, dtype=x.dtype)
        B22_diag = self.diag.unsqueeze(-1) * eye  
        
        # Low Rank part (Woodbury correction)
        # (The 'k' in this einsum string handles self.k, i.e., num_bcs)
        B22_lr = torch.einsum('iotk, ioSk -> iotS', self.U_lr, self.V_lr)
        
        # Complete B22
        B22 = B22_semi + B22_diag + B22_lr

        # --- Stitch the full Matrix W ---
        # W = [ B11   B12 ]
        #     [ B21   B22 ]
        # print(self.B11.shape, self.B12.shape, self.B21.shape, B22.shape)
        # assert False
        row1 = torch.cat([self.B11, self.B12], dim=-1)
        row2 = torch.cat([self.B21, B22], dim=-1)      
        W = torch.cat([row1, row2], dim=-2)            

        # --- Apply to Input ---
        y = torch.einsum('iotS, biS -> bot', W, x)
        
        return y

    def forward(self, x):
        # x : (batches, nx, features)
        batch_size, channels, Nx = x.shape

        b = dctn(x, -1)
        out = torch.zeros(
            batch_size, self.out_c, Nx, device=x.device, dtype=torch.float32
        )

        out[..., : self.modes] = self.ultraspherical_forward(b[..., : self.modes])

        u = idctn(out, -1)

        return u 



class SSUltraNet1D(nn.Module):
    def __init__(self, modes, width, rank, n_bc: int):
        super(SSUltraNet1D, self).__init__()
        self.degree = modes
        self.width = width
        self.rank = rank
        self.n_bc = n_bc

        self.conv0 = UltrasphericalInverse(in_c=self.width, out_c=self.width, modes=self.degree, num_bcs=self.n_bc, semi_rank=self.rank)
        self.conv1 = UltrasphericalInverse(in_c=self.width, out_c=self.width, modes=self.degree, num_bcs=self.n_bc, semi_rank=self.rank)
        self.conv2 = UltrasphericalInverse(in_c=self.width, out_c=self.width, modes=self.degree, num_bcs=self.n_bc, semi_rank=self.rank)
        self.conv3 = UltrasphericalInverse(in_c=self.width, out_c=self.width, modes=self.degree, num_bcs=self.n_bc, semi_rank=self.rank)

        self.convl = UltrasphericalInverse(in_c=2, out_c=self.width-2, modes=self.degree, num_bcs=self.n_bc, semi_rank=self.rank)

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

        return x


if __name__ == "__main__":
    x = torch.rand(10, 12, 2)

    # learnable_qii = LearnableQII(32, 32, 12, 4, 4)
    # new_ultra = UltrasphericalInverse(1, 1, 12, 4, 4)
    # print(new_ultra(x).shape)
    # print(sum(p.numel() for p in new_ultra.parameters()))

    ss = SSUltraNet1D(12, 10, 4, 4)
    print(ss(x).shape)
    print(sum(p.numel() for p in ss.parameters()))



