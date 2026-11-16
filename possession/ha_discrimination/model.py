"""Physics concatenation or matched-capacity residual; frozen teacher output."""
import torch
from torch import nn
from torch.nn import functional as F


class PhysicalHead(nn.Module):
    def __init__(self, original, geometry=False, width=64, seed=2027):
        super().__init__()
        self.intervals=False; self.temporal=False; self.context=0
        self.use_geometry=geometry
        self.register_buffer("team_weight",original.team_weight.clone())
        self.register_buffer("team_bias",original.team_bias.clone())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+2)
            self.hidden=nn.Linear(84 if geometry else 64,width)
            self.team=nn.Linear(width,1)
            with torch.no_grad():
                self.hidden.weight[:64,:64].copy_(original.hidden.weight)
                self.hidden.bias[:64].copy_(original.hidden.bias)
                if geometry:self.hidden.weight[:,64:].zero_()
                self.team.weight.zero_(); self.team.bias.zero_()

    def forward(self,hidden,geometry):
        x=torch.cat((hidden,geometry),-1) if self.use_geometry else hidden
        residual=self.team(F.relu(self.hidden(x))).squeeze(-1)
        return hidden@self.team_weight+self.team_bias+residual,None,None
