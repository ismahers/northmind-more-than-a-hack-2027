"""Four matched-schedule heads; no trainable alive/dead component."""
import torch
from torch import nn
from torch.nn import functional as F

from possession.interval_probe.model import interval_loss


class TemporalHead(nn.Module):
    def __init__(self, team_weight, team_bias, temporal=False, intervals=False,
                 dilations=(1, 4, 16), kernel_size=5, seed=2027):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 != 1 or not dilations or min(dilations) < 1:
            raise ValueError("Odd kernel and positive dilations required")
        self.temporal, self.intervals = temporal, intervals
        self.context = max(dilations) * (kernel_size // 2) if temporal else 0
        self.register_buffer("team_weight", team_weight.detach().reshape(64).clone())
        self.register_buffer("team_bias", team_bias.detach().reshape(()).clone())
        # Separate initialization stream keeps shared parameters identical in all arms.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.hidden = nn.Linear(64, 64)
            self.team = nn.Linear(64, 1)
            nn.init.zeros_(self.team.weight)
            nn.init.zeros_(self.team.bias)
            if intervals:
                self.location = nn.Linear(64, 6)
        if temporal:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 1)
                self.filters = nn.ModuleList([
                    nn.Conv1d(64, 64, kernel_size, padding=d * (kernel_size // 2),
                              dilation=d, groups=64) for d in dilations])
                self.mix = nn.Linear(64 * len(dilations), 64)

    def forward(self, hidden):
        if hidden.ndim != 3 or hidden.shape[-1] != 64:
            raise ValueError("Expected batch, time, 64 embeddings")
        value = self.hidden(hidden)
        if self.temporal:
            scales = [f(hidden.transpose(1, 2)).transpose(1, 2) for f in self.filters]
            value = value + self.mix(torch.cat(scales, dim=-1))
        value = F.relu(value)
        team = hidden @ self.team_weight + self.team_bias + self.team(value).squeeze(-1)
        if not self.intervals:
            return team, None, None
        output = self.location(value)
        return team, 3 * F.softplus(output[..., :4].reshape(*team.shape, 2, 2)), output[..., 4:]


def loss_for_core(output, target, valid, intervals):
    """Halo/padding never contribute; D and unknown never supervise team."""
    team, distance, quality = output
    if intervals:
        if not valid.any():
            return team.sum() * 0
        core = {name: value[valid] for name, value in target.items()}
        live = (core["label"] >= 0) & (core["label"] < 2)
        # The historical loss expects complete labels. Unknowns carry zero weight.
        core["condition_weight"] = core["condition_weight"] * (core["label"] >= 0)
        core["regression_valid"] = core["regression_valid"] & live
        core["label"] = torch.where(core["label"] < 0, 2, core["label"])
        return interval_loss((team[valid], distance[valid], quality[valid]), core, "diou")[0]
    live = valid & (target["label"] >= 0) & (target["label"] < 2)
    weight = target["condition_weight"] * live
    error = F.binary_cross_entropy_with_logits(team, (target["label"] == 0).float(), reduction="none")
    return (error * weight).sum() / weight.sum().clamp_min(1)
