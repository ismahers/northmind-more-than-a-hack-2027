"""Match attention parameter count while retaining independent-player pooling."""
import torch
from torch import nn
from possession.phase7.model import SpatialE0


class CapacityE0(SpatialE0):
    def __init__(self,auxiliary=False):
        super().__init__("harmonized",auxiliary)
        state=torch.get_rng_state()
        self.encoder=nn.Sequential(nn.Linear(6,96),nn.ReLU(),nn.Linear(96,128),nn.ReLU(),nn.Linear(128,32),nn.ReLU())
        torch.set_rng_state(state)
