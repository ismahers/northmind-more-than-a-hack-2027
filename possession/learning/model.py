"""E0: shared DeepSets per team, symmetric-context dilated TCN, factorized heads."""

import torch
from torch import nn
from torch.nn import functional as F

from possession.external_data.features import GLOBAL_DIM, NODE_DIM


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.project = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, valid):
        y = self.project(self.dropout(F.relu(self.conv(x))))
        return (x + y) * valid


class E0(nn.Module):
    def __init__(self, spatial_dim=32, channels=64, dilations=(1, 2, 4, 8), dropout=0.1):
        super().__init__()
        if not dilations or any(d < 1 or int(d) != d for d in dilations):
            raise ValueError("Positive integer dilations required")
        self.context = sum(dilations)  # One kernel-3 convolution per residual block, each side.
        self.encoder = nn.Sequential(nn.Linear(NODE_DIM, spatial_dim), nn.ReLU(),
                                     nn.Linear(spatial_dim, spatial_dim), nn.ReLU())
        self.input = nn.Conv1d(4 * spatial_dim + GLOBAL_DIM, channels, 1)
        self.blocks = nn.ModuleList([ResidualTemporalBlock(channels, d, dropout) for d in dilations])
        self.alive_head = nn.Conv1d(channels, 1, 1)
        self.team_head = nn.Conv1d(channels, 1, 1)

    def forward(self, nodes, node_mask, global_features, valid):
        # [B,T,2,P,F]. Pool separately by team; no player index/identity enters the network.
        h = self.encoder(nodes)
        weights = node_mask.unsqueeze(-1)
        mean = (h * weights).sum(-2) / weights.sum(-2).clamp_min(1)
        maximum = h.masked_fill(~weights, -torch.inf).amax(-2)
        maximum = torch.where(node_mask.any(-1, keepdim=True), maximum, torch.zeros_like(maximum))
        pooled = torch.cat((mean, maximum), -1).flatten(-2)
        valid_channel = valid.unsqueeze(1).to(nodes.dtype)
        x = self.input(torch.cat((pooled, global_features), -1).transpose(1, 2)) * valid_channel
        for block in self.blocks:
            x = block(x, valid_channel)
        alive = self.alive_head(x).squeeze(1)
        team = self.team_head(x).squeeze(1)  # P(home|alive), not P(away).
        a, h = alive.sigmoid(), team.sigmoid()
        probabilities = torch.stack((a * h, a * (1 - h), 1 - a), -1)
        return {"alive_logit": alive, "team_logit": team, "probabilities": probabilities}


def masked_losses(output, alive, team, confidence, valid):
    state_weight = confidence[..., 0] * valid * (alive >= 0)
    # SkillCorner may supervise conditional team while state remains unknown.
    team_weight = confidence[..., 1] * valid * (team >= 0) * (alive != 0)
    state_error = F.binary_cross_entropy_with_logits(output["alive_logit"], alive.clamp_min(0).float(), reduction="none")
    team_error = F.binary_cross_entropy_with_logits(output["team_logit"], (team == 0).float(), reduction="none")
    state_sum, team_sum = (state_error * state_weight).sum(), (team_error * team_weight).sum()
    state_n, team_n = state_weight.sum(), team_weight.sum()
    loss = state_sum / state_n.clamp_min(1) + team_sum / team_n.clamp_min(1)
    return loss, torch.stack((state_sum.detach(), state_n.detach(), team_sum.detach(), team_n.detach()))
