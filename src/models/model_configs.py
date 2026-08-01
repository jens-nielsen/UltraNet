
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
    modes: int
    width: int

    def create_model(self, d: int) -> torch.nn.Module:
        if d == 1:
            from .fno import FNO1d
            return FNO1d(self.modes, self.width)
    

# OPNO model
@dataclass
class OPNOConfig(ModelConfig):
    modes: int
    width: int

    def create_model(self, d: int) -> torch.nn.Module:
        if d == 1:
            from .opno import OPNO
            return OPNO(self.modes, self.width)
        
# UltraNet model
@dataclass
class UltraNetConfig(ModelConfig):
    modes: int
    width: int
    rank: int

    def create_model(self, d: int) -> torch.nn.Module:
        if d == 1:
            from .ultranet import UltraNet
            return UltraNet(self.modes, self.width, self.rank)


class ModelType(Enum):
    FNO = "fno"
    OPNO = "opno"
    UltraNet = "ultranet"

model_config_mapping: dict[ModelType, ModelConfig] = {
    ModelType.FNO: FNOConfig,
    ModelType.OPNO: OPNOConfig,
    ModelType.UltraNet: UltraNetConfig,
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