
from dataclasses import dataclass
from typing import Literal

import h5py
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import json
from enum import Enum

#################################################
#
# Utilities
#
#################################################


# reading data
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data: dict | h5py.File = {}
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float



class BoundaryType(Enum):
    PERIODIC = "p"
    NEUMANN = "n"
    DIRICHLET = "d"
    DIRICHLET_LEFT = "d_left"
    ROBIN = "r"
@dataclass
class DataConfig:
    pth: str
    cheby_pth: str | None
    d: int
    o_d: int = 1 # Output dimensionality
    bc: BoundaryType = None


BurgersData = DataConfig(
    pth="./datasets/burgers_neumann_uniform.pt",
    cheby_pth="./datasets/burgers_neumann_chebyshev.pt",
    d=1,
    o_d=1,
    bc=BoundaryType.NEUMANN

)
BurgersPeriodicData = DataConfig(
    pth="./datasets/burgers_periodic_uni.pt",
    cheby_pth="./datasets/burgers_periodic_cgl.pt",
    d=1,
    o_d=1,
    bc=BoundaryType.DIRICHLET
)
BurgersPeriodicPlusNeumannData = DataConfig(
    pth="./datasets/burgers_nplusp_uni.pt",
    cheby_pth="./datasets/burgers_nplusp_cgl.pt",
    d=1,
    o_d=1,
    bc=None
)

BurgersPSRobinData = DataConfig(
    pth="./datasets/burgers_psrobin_uni.pt",
    cheby_pth="./datasets/burgers_psrobin_cgl.pt",
    d=1,
    o_d=1,
    bc=BoundaryType.ROBIN
)
BurgersPSRobinSimpleData = DataConfig(
    pth="./datasets/burgers_psrobin_simple_uni.pt",
    cheby_pth="./datasets/burgers_psrobin_simple_cgl.pt",
    d=1,
    o_d=1,
    bc=BoundaryType.ROBIN
)

DarcyNeumannData = DataConfig(
    pth="./datasets/darcy_neumann_uni.pt",
    cheby_pth="./datasets/darcy_neumann_cgl.pt",
    d=1,
    o_d=1,
    bc=BoundaryType.NEUMANN
)
DarcyNeumannAData = DataConfig(
    pth="./datasets/darcy_neumann_a_uni.pt",
    cheby_pth="./datasets/darcy_neumann_a_cgl.pt",
    d=1,
    o_d=1,
    bc=BoundaryType.NEUMANN
)
DarcyData = DataConfig(
    pth="./datasets/darcy_flow_uniform.pt",
    cheby_pth="./datasets/darcy_flow_chebyshev.pt",
    d=2,
    o_d=1,
    bc=BoundaryType.DIRICHLET
)

HelmholtzData = DataConfig(
    pth="./datasets/helmholtz_uni.pt",
    cheby_pth="./datasets/helmholtz_cgl.pt",
    d=1,
    o_d=2,
)

Helmholtz2DData = DataConfig(
    pth="./datasets/Helmholtz2D_uni.pt",
    cheby_pth="./datasets/Helmholtz2D_cgl.pt",
    d=2,
    o_d=2
)

# ODE TEST

ODE1DData1 = DataConfig(
    pth="./datasets/ode1D_1_uniform.pt",
    cheby_pth="./datasets/ode1D_1_chebyshev.pt",
    d=1,
    o_d=1,
    bc=BoundaryType.DIRICHLET_LEFT
)
ODE1DData2 = DataConfig(
    pth="./datasets/ode1D_2_uniform.pt",
    cheby_pth="./datasets/ode1D_2_chebyshev.pt",
    d=1,
    o_d=1,
    bc=BoundaryType.DIRICHLET_LEFT
)
ODE1DData4 = DataConfig(
    pth="./datasets/ode1D_4_uniform.pt",
    cheby_pth="./datasets/ode1D_4_chebyshev.pt",
    d=1,
    o_d=1,
    bc=BoundaryType.DIRICHLET_LEFT
)
ODE1DData8 = DataConfig(
    pth="./datasets/ode1D_8_uniform.pt",
    cheby_pth="./datasets/ode1D_8_chebyshev.pt",
    d=1,
    o_d=1,
    bc=BoundaryType.DIRICHLET_LEFT
)
ODE1DData16 = DataConfig(
    pth="./datasets/ode1D_16_uniform.pt",
    cheby_pth="./datasets/ode1D_16_chebyshev.pt",
    d=1,
    o_d=1,
    bc=BoundaryType.DIRICHLET_LEFT
)
ODE1DData32 = DataConfig(
    pth="./datasets/ode1D_32_uniform.pt",
    cheby_pth="./datasets/ode1D_32_chebyshev.pt",
    d=1,
    o_d=1,
    bc=BoundaryType.DIRICHLET_LEFT
)
ODE1DData64 = DataConfig(
    pth="./datasets/ode1D_64_uniform.pt",
    cheby_pth="./datasets/ode1D_64_chebyshev.pt",
    d=1,
    o_d=1,
    bc=BoundaryType.DIRICHLET_LEFT
)


ODE1DData2_ = DataConfig(
    pth="./datasets/ode1D_2_uniform_2.pt",
    cheby_pth="./datasets/ode1D_2_chebyshev_2.pt",
    d=1,
    o_d=1,
    bc=BoundaryType.DIRICHLET_LEFT
)

class DataType(Enum):
    BURGERS = "burgers"
    BURGERS_PERIODIC = "burgers_p"
    BURGERS_PERIODIC_NEUMANN = "burgers_npp"
    BURGERS_PSROBIN = "burgers_psr"
    BURGERS_PSROBIN_SIMPLE = "burgers_psr_s"
    DARCY_NEUMANN = "darcy_n"
    DARCY_NEUMANN_A = "darcy_n_a"
    DARCY = "darcy"
    HELMHOLTZ = "helmholtz"
    HELMHOLTZ_2D= "helmholtz2D"


    ODE1D1="ode1"
    ODE1D2="ode2"
    ODE1D4="ode4"
    ODE1D8="ode8"
    ODE1D16="ode16"
    ODE1D32="ode32"
    ODE1D64="ode64"


    ODE1D2_="ode2"


data_config_map: dict[DataType, DataConfig] = {
    DataType.BURGERS: BurgersData,
    DataType.BURGERS_PERIODIC: BurgersPeriodicData,
    DataType.BURGERS_PSROBIN: BurgersPSRobinData,
    DataType.BURGERS_PERIODIC_NEUMANN: BurgersPeriodicPlusNeumannData,
    DataType.BURGERS_PSROBIN_SIMPLE: BurgersPSRobinSimpleData,
    DataType.DARCY_NEUMANN: DarcyNeumannData,
    DataType.DARCY_NEUMANN_A: DarcyNeumannAData,
    DataType.DARCY: DarcyData,
    DataType.HELMHOLTZ: HelmholtzData,
    DataType.HELMHOLTZ_2D: Helmholtz2DData,

    DataType.ODE1D1: ODE1DData1,
    DataType.ODE1D2: ODE1DData2,
    DataType.ODE1D4: ODE1DData4,
    DataType.ODE1D8: ODE1DData8,
    DataType.ODE1D16: ODE1DData16,
    DataType.ODE1D32: ODE1DData32,
    DataType.ODE1D64: ODE1DData64,


    DataType.ODE1D2_: ODE1DData2_,
}

def generate_grid(sizes: int | list[int], cheby: bool) -> torch.Tensor:

    def generate_points(size: int, cheby: bool) -> torch.Tensor:
        if cheby:
            x = torch.cos(torch.pi * torch.linspace(1, 0, size))
        else:
            x = torch.linspace(0, 1, size, dtype=torch.float32)

        return x
    
    if isinstance(sizes, int):
        sizes = [sizes]

    points = []

    for s in sizes:
        points.append(generate_points(s, cheby))

    grids = torch.meshgrid(*points, indexing="ij")

    return torch.stack(grids, dim=-1)

def sample_at_2d_custom_grid(tensor_3d, custom_grid, mode="bicubic"):
        """
        Samples a (B, H, W) tensor at specific coordinates provided in custom_grid.

        Args:
            tensor_3d: Input data of shape (B, H_in, W_in)
            custom_grid: Coordinates of shape (1, H_out, W_out, 2)
                         Values MUST be in range [-1, 1].
                         Last dim is (x, y) where x is horizontal, y is vertical.
            mode: 'bilinear', 'bicubic', or 'nearest'

        Returns:
            Sampled tensor of shape (B, H_out, W_out)
        """
        # 1. Ensure input is 4D: (B, 1, H_in, W_in)
        x = tensor_3d.unsqueeze(1)

        # 1.5 Add batch dimension to custom_grid: (B, H_out, W_out, 2)
        custom_grid = custom_grid.expand(x.size(0), -1, -1, -1)

        # 2. Apply grid_sample
        # padding_mode='border' is usually best for physics to avoid zero-leaks at edges
        sampled = torch.nn.functional.grid_sample(
            x, custom_grid, mode=mode, padding_mode="border", align_corners=True
        )

        # 3. Return to 3D: (B, H_out, W_out)
        return sampled.squeeze(1)

class NeuralOperatorDataset:
    def __init__(self, data: DataType, is_cheby: bool, ntrain: int, ntest: int, batch_size: int, normalize: bool, subsample: int):
        self.ntrain = ntrain
        self.ntest = ntest
        self.batch_size = batch_size
        self.chebyshev = is_cheby
        self.data_type = data
        self.normalize = normalize


        # reader = MatReader("./datasets/burgers_neumann.mat")
        # a = reader.read_field("u0_cgl").permute(1, 0)
        # u = reader.read_field("u1_cgl").permute(1, 0)
        # a_train = a[:ntrain][..., ::subsample].unsqueeze(-1)
        # u_train = u[:ntrain][..., ::subsample].unsqueeze(-1)
        # a_test = a[-ntest:][..., ::subsample].unsqueeze(-1)
        # u_test = u[-ntest:][..., ::subsample].unsqueeze(-1)
        self.cfg = data_config_map.get(data)
        self.d = self.cfg.d

        pth = self.cfg.cheby_pth if is_cheby else self.cfg.pth
        data = torch.load(pth)
        # print(data["X"].shape)
        # data_uni = {
        #     "a": data["X"].permute(0, 2, 3, 1),
        #     "u": data["Y"].permute(0, 2, 3, 1)
        # }
        # data_cgl = {
        #     "a": data["X_cheb"].permute(0, 2, 3, 1),
        #     "u": data["Y_cheb"].permute(0, 2, 3, 1)
        # }
        # torch.save(data_uni, "./datasets/Helmholtz2D_uni.pt")
        # torch.save(data_cgl, "./datasets/Helmholtz2D_cgl.pt")
        # assert False
        subsample_index = [slice(None, None, subsample),] * self.d
        a_train = data['a'][:ntrain][:, *subsample_index].to(torch.float32)
        u_train = data['u'][:ntrain][:, *subsample_index].to(torch.float32)
        a_test = data['a'][-ntest:][:, *subsample_index].to(torch.float32)
        u_test = data['u'][-ntest:][:, *subsample_index].to(torch.float32)

        # # Normalize data
        if normalize:
            print("Normalizing data...")
            self.eps = 1e-7
            self.a_mean, self.a_std = torch.mean(a_train, axis=0, keepdim=True), torch.std(a_train, axis=0, keepdim=True)
            self.u_mean, self.u_std = torch.mean(u_train, axis=0, keepdim=True), torch.std(u_train, axis=0, keepdim=True)
            a_train = (a_train - self.a_mean) / (self.a_std + self.eps)
            u_train = (u_train - self.u_mean) / (self.u_std + self.eps)
            a_test = (a_test - self.a_mean) / (self.a_std + self.eps)
            u_test = (u_test - self.u_mean) / (self.u_std + self.eps)

        # Add positional encodings
        grid = generate_grid(a_train.shape[1:(1+self.cfg.d)], is_cheby)
        a_train = torch.cat([a_train, # Add final dimension
                             grid.expand(a_train.shape[0], *grid.shape)], dim=-1)
        a_test = torch.cat([a_test, # Add final dimension
                            grid.expand(a_test.shape[0], *grid.shape)], dim=-1)

        self.train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(a_train, u_train), batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(a_test, u_test), batch_size=self.batch_size, shuffle=False)

if __name__ == "__main__":
    #  Test the dataset class
    data = NeuralOperatorDataset(DataType.HELMHOLTZ_2D, is_cheby=False, ntrain=1000, ntest=200, batch_size=32, normalize=True, subsample=1)

    # data_cgl = torch.load("./datasets/burgers_periodic_cgl.pt")
    # data_uni = torch.load("./datasets/burgers_periodic_uni.pt")

    # print(data_cgl["a"].shape)
    # data_cgl_update = {
    #     "u": data_cgl["u"].unsqueeze(-1),
    #     "a": data_cgl["a"].unsqueeze(-1),
    #     "x": data_cgl["x"]
    # }

    # data_uni_update = {
    #     "u": data_uni["u"].unsqueeze(-1),
    #     "a": data_uni["a"].unsqueeze(-1),
    #     "x": data_uni["x"]
    # }

    # print(data_cgl_update["a"].shape)
    # torch.save(data_cgl_update, "./datasets/burgers_periodic_cgl.pt")
    # torch.save(data_uni_update, "./datasets/burgers_periodic_uni.pt")


    
