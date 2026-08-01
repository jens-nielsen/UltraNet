
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


@dataclass
class DataConfig:
    pth: str
    cheby_pth: str | None
    d: int

BurgersData = DataConfig(
    pth="./datasets/burgers_data_uniform_R10.pt",
    cheby_pth="./datasets/burgers_data_chebyshev_R10.pt",
    d=1,
)
DarcyData = DataConfig(
    pth="./datasets/darcy_flow_uniform.pt",
    cheby_pth="./datasets/darcy_flow_chebyshev.pt",
    d=2,
)
HelmholtzData = DataConfig(
    pth="./datasets/Helmholtz_10000_128.pt",
    cheby_pth=None,
    d=2,
)

class DataType(Enum):
    BURGERS = "burgers"
    DARCY = "darcy"
    HELMHOLTZ = "helmholtz"

data_config_map: dict[DataType, DataConfig] = {
    DataType.BURGERS: BurgersData,
    DataType.DARCY: DarcyData,
    DataType.HELMHOLTZ: HelmholtzData,
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

class NeuralOperatorDataset:
    def __init__(self, data: DataType, is_cheby: bool, ntrain: int, ntest: int, batch_size: int, subsample: int = 1):
        self.ntrain = ntrain
        self.ntest = ntest
        self.batch_size = batch_size
        self.chebyshev = is_cheby
        self.data_type = data

        self.cfg = data_config_map.get(data)
        self.d = self.cfg.d

        pth = self.cfg.cheby_pth if is_cheby else self.cfg.pth
        data = torch.load(pth)

        subsample_index = [slice(None, None, subsample),] * self.d
        a_train = data['a'][:ntrain][:, *subsample_index]
        u_train = data['u'][:ntrain][:, *subsample_index]
        a_test = data['a'][-ntest:][:, *subsample_index]
        u_test = data['u'][-ntest:][:, *subsample_index]

        # Normalize data
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

        print(a_train.shape, u_train.shape, a_test.shape, u_test.shape)

        self.train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(a_train, u_train), batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(a_test, u_test), batch_size=self.batch_size, shuffle=False)



if __name__ == "__main__":
    # Test the dataset class
    data = NeuralOperatorDataset(DataType.DARCY, is_cheby=False, ntrain=1000, ntest=200, batch_size=32)

