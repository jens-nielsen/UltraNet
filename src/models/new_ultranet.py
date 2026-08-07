import torch
import torch.nn as nn
from . import chebypack as ch
import functools


idctn = functools.partial(ch.Wrapper, [ch.idct])
dctn = functools.partial(ch.Wrapper, [ch.dct])

class LearnableSemiseparable(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int, rank: int):
        super().__init__()
        # Learnable generators for lower (P, Q) and upper (U, V) triangular parts
        self.scale = 2/(in_channels + out_channels)
        self.P = nn.Parameter(torch.randn(in_channels, out_channels, modes, rank) * self.scale)
        self.Q = nn.Parameter(torch.randn(in_channels, out_channels, modes, rank) * self.scale)
        self.U = nn.Parameter(torch.randn(in_channels, out_channels, modes, rank) * self.scale)
        self.V = nn.Parameter(torch.randn(in_channels, out_channels, modes, rank) * self.scale)
        self.diag = nn.Parameter(torch.randn(in_channels, out_channels, modes)* self.scale)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        b, i, n = x.shape
        
        # 1. Lower triangular part (including diagonal): tril(P @ Q.T) @ x
        # Q_x is (batch_size, seq_len, rank)
        Q_x = self.Q.unsqueeze(0) * x.unsqueeze(-1) 
        # Cumulative sum calculates the causal prefix sum in O(N)
        cumsum_Q_x = torch.cumsum(Q_x, dim=1) 
        lower_y = (self.P.unsqueeze(0) * cumsum_Q_x).sum(dim=-1)
        
        # 2. Upper triangular part (strictly upper): triu(U @ V.T, 1) @ x
        V_x = self.V.unsqueeze(0) * x.unsqueeze(-1)
        # Reverse cumulative sum for the anti-causal part
        rev_cumsum_V_x = torch.flip(torch.cumsum(torch.flip(V_x, dims=[1]), dim=1), dims=[1])
        # Shift by 1 to make it strictly upper triangular
        shifted_rev_cumsum = torch.cat([rev_cumsum_V_x[:, 1:, :], torch.zeros_like(rev_cumsum_V_x[:, :1, :])], dim=1)
        upper_y = (self.U.unsqueeze(0) * shifted_rev_cumsum).sum(dim=-1)
        
        # 3. Add diagonal correction if needed (or absorb into lower/upper)
        return lower_y + upper_y + self.diag * x

class LearnableLowRank(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int, rank: int):
        super().__init__()
        # Generators U and V of shape (seq_len, rank)
        # We scale the initialization to keep variances stable
        self.scale = 2/(in_channels + out_channels)
        self.U = nn.Parameter(torch.randn(in_channels, out_channels, modes, rank) * self.scale)
        self.V = nn.Parameter(torch.randn(in_channels, out_channels, modes, rank) * self.scale)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len)
        Returns:
            y: Tensor of shape (batch_size, seq_len)
        """
        # Step 1: Compute V^T * x 
        # (batch_size, seq_len) @ (seq_len, rank) -> (batch_size, rank)
        # This is equivalent to taking the dot product of x with every column of V
        c = torch.einsum('bin, ionr -> bior', x, self.V)
        y = torch.einsum('bior, ionr -> bon', c, self.U)
        return y


import torch
import torch.nn as nn

class MultiChannelSemiseparableExplicit(nn.Module):
    def __init__(self, in_c: int, out_c: int, modes: int, rank: int):
        super().__init__()
        # Generators remain the same shape: (in_c, out_c, seq_len, rank)
        self.scale = 2 / (in_c + out_c)
        self.P = nn.Parameter(torch.randn(in_c, out_c, modes, rank) * self.scale)
        self.Q = nn.Parameter(torch.randn(in_c, out_c, modes, rank) * self.scale)
        
        self.U = nn.Parameter(torch.randn(in_c, out_c, modes, rank) * self.scale)
        self.V = nn.Parameter(torch.randn(in_c, out_c, modes, rank) * self.scale)
        
        self.diag = nn.Parameter(torch.randn(in_c, out_c, modes))

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, in_c, seq_len)
        Returns:
            y: Tensor of shape (batch_size, out_c, seq_len)
        """
        # --- 1. Form the T x T matrices independent of the batch ---
        # P @ Q^T -> shape: (in_c, out_c, seq_len, seq_len)
        # We use 'S' for the second sequence dimension
        lower_dense = torch.einsum('iotk, ioSk -> iotS', self.P, self.Q)
        
        # Apply causal mask (lower triangular)
        lower_masked = torch.tril(lower_dense)
        
        # U @ V^T -> shape: (in_c, out_c, seq_len, seq_len)
        upper_dense = torch.einsum('iotk, ioSk -> iotS', self.U, self.V)
        
        # Apply strictly anti-causal mask (upper triangular, shifted by 1)
        upper_masked = torch.triu(upper_dense, diagonal=1)
        
        # --- 2. Combine the masks ---
        # W shape: (in_c, out_c, seq_len, seq_len)
        W = lower_masked + upper_masked
        
        # --- 3. Apply to the batched input x ---
        # W @ x -> sum over in_c (i) and input seq_len (S)
        # W is (i, o, t, S) and x is (b, i, S) -> y is (b, o, t)
        y = torch.einsum('iotS, biS -> bot', W, x)
        
        # --- 4. Diagonal Correction ---
        y_diag = torch.einsum('iot, bit -> bot', self.diag, x)
        
        return y + y_diag

class LearnableQII(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int, semiseparable_rank: int, boundary_rank: int):
        super().__init__()
        self.semiseparable = MultiChannelSemiseparableExplicit(in_c=in_channels, out_c=out_channels, modes=modes, rank=semiseparable_rank)
        self.lr_correction = LearnableLowRank(in_channels=in_channels, out_channels=out_channels, modes=modes, rank=boundary_rank)

    def forward(self, x):
        x_ss = self.semiseparable(x)
        x_lrc = self.lr_correction(x)
        return x_ss + x_lrc
 
import torch
import torch.nn as nn

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

        print(self.interior_len, semi_rank)
        
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
    def __init__(self, modes, width, rank, n_bc):
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



