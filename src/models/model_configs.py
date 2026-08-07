
from typing import Literal
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


@dataclass
class ModelConfig(ABC):
    @abstractmethod
    def create_model(self, d: int) -> torch.nn.Module:
        raise NotImplementedError("Must implement create method in subclass.")

# FNO model
@dataclass
class FNOConfig(ModelConfig):
    modes: int | list[int]
    width: int

    def create_model(self, d: int) -> torch.nn.Module:
        if d == 1:
            from .fno import FNO1d
            return FNO1d(modes1=self.modes, width=self.width)
        elif d == 2:
            from .fno import FNO2d
            return FNO2d(modes1=self.modes[0], modes2=self.modes[1], width=self.width)
    

# OPNO model
@dataclass
class OPNOConfig(ModelConfig):
    modes: int | list[int]
    width: int

    def create_model(self, d: int) -> torch.nn.Module:
        if d == 1:
            from .opno import OPNO1D
            return OPNO1D(self.modes, self.width)
        elif d == 2:
            from .opno import OPNO2d
            return OPNO2d(degree1=self.modes[0], degree2=self.modes[1], width=self.width)
        
# UltraNet model
@dataclass
class UltraNetConfig(ModelConfig):
    modes: int | list[int]
    width: int
    rank: int

    def create_model(self, d: int) -> torch.nn.Module:
        if d == 1:
            from .ultranet import UltraNet1D
            return UltraNet1D(self.modes, self.width, self.rank)
        elif d == 2:
            from .ultranet import UltraNet2D
            return UltraNet2D(degree1=self.modes[0], degree2=self.modes[1], width=self.width, rank=self.rank)

# SSUltraNet model
# 
@dataclass
class SSUltraNetConfig(ModelConfig):
    modes: int | list[int]
    width: int
    rank: int
    nbc: int

    def create_model(self, d: int) -> torch.nn.Module:
        if d == 1:
            from .new_ultranet import SSUltraNet1D
            return SSUltraNet1D(modes=self.modes, width=self.width, rank=self.rank, n_bc=self.nbc)
        
 

class ModelType(Enum):
    FNO = "fno"
    OPNO = "opno"
    UltraNet = "ultranet"
    SSUltraNet = "ss"

model_config_mapping: dict[ModelType, ModelConfig] = {
    ModelType.FNO: FNOConfig,
    ModelType.OPNO: OPNOConfig,
    ModelType.UltraNet: UltraNetConfig,
    ModelType.SSUltraNet: SSUltraNetConfig,
}

class NeuralOperatorModel(nn.Module):
    def __init__(self, model: ModelType, d: int, **kwargs):
        super().__init__()
        self.model_config: ModelConfig = model_config_mapping[model](**kwargs)
        self.model = self.model_config.create_model(d)

    def forward(self, x):
        return self.model(x)    

    # print the number of parameters
    def count_params(self):
        return sum(p.numel() for p in self.model.parameters())