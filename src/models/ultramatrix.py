from src.utils.data import BoundaryType

from .ultranet import BPSPseudoSpectra
import torch

class UltraMatrix(torch.nn.Module):
    def __init__(self, in_dim, out_dim, modes, rank, bc: BoundaryType):
        super(UltraMatrix, self).__init__()
        self.matrix = BPSPseudoSpectra(in_dim, out_dim, v_modes=modes, bandwidth=1, rank=rank, bc=bc)


    def forward(self, x):
        x = x[..., 0:1] # Ignore posiitonal encodings
        x = x.permute(0, 2, 1)
        x = self.matrix(x)
        x = x.permute(0, 2, 1)
        return x