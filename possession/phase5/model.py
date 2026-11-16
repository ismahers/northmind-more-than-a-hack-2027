"""E0 with one auxiliary head; no label is an input to forward/inference."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from possession.external_data.audit import joint_labels, segments
from possession.learning.model import E0
from possession.learning.data import Windows, collate


class BoundaryE0(E0):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.boundary_head = nn.Conv1d(self.team_head.in_channels, 1, 1)

    def forward(self, nodes, node_mask, global_features, valid):
        # Identical shared trunk and existing heads to E0. Symmetric convolutions
        # see 1.5 seconds on each side; the new head adds only 65 parameters.
        h = self.encoder(nodes)
        weights = node_mask.unsqueeze(-1)
        mean = (h * weights).sum(-2) / weights.sum(-2).clamp_min(1)
        maximum = h.masked_fill(~weights, -torch.inf).amax(-2)
        maximum = torch.where(node_mask.any(-1, keepdim=True), maximum, torch.zeros_like(maximum))
        pooled = torch.cat((mean, maximum), -1).flatten(-2)
        mask = valid.unsqueeze(1).to(nodes.dtype)
        x = self.input(torch.cat((pooled, global_features), -1).transpose(1, 2)) * mask
        for block in self.blocks:
            x = block(x, mask)
        alive, team = self.alive_head(x).squeeze(1), self.team_head(x).squeeze(1)
        a, h = alive.sigmoid(), team.sigmoid()
        return {"alive_logit": alive, "team_logit": team,
                "probabilities": torch.stack((a * h, a * (1 - h), 1 - a), -1),
                "boundary_logit": self.boundary_head(x).squeeze(1)}


def boundary_targets(match, view, tolerance=.12):
    """Native direct H<->A events only; D/restarts never become recoveries.

    Unknown labels censor the tolerance neighbourhood. Run endpoints cannot
    create events. Binary positive bands tolerate at most tolerance seconds;
    they do not invent native labels or cross a dead-ball interval.
    """
    labels = joint_labels(match)
    target, valid = np.zeros(len(view.time), np.float32), np.zeros(len(view.time), bool)
    for run_id, (a, b) in enumerate(segments(match)):
        ix = np.flatnonzero(view.run == run_id)
        times = view.time[ix]
        y, t = labels[a:b], match.time[a:b]
        events = t[1:][(y[:-1] < 2) & (y[:-1] >= 0) & (y[1:] < 2) & (y[1:] >= 0) & (y[1:] != y[:-1])]
        known = np.ones(len(ix), bool)
        for bad in t[y < 0]:
            known &= np.abs(times - bad) > tolerance
        positive = np.zeros(len(ix), bool)
        # searchsorted avoids a dense events-by-frames temporary.
        for event in events:
            lo, hi = np.searchsorted(times, [event - tolerance, event + tolerance], side="left")
            positive[lo:hi] = True
        target[ix] = positive & (view.alive[ix] == 1) & (view.team[ix] >= 0)
        valid[ix] = known & (view.alive[ix] >= 0) & ((view.alive[ix] == 0) | (view.team[ix] >= 0))
    return target, valid


class BoundaryWindows(Windows):
    def __init__(self, views, targets, core, context):
        super().__init__(views, core, context)
        self.targets = targets

    def __getitem__(self, i):
        item = super().__getitem__(i)
        key, left, right, _, _ = self.windows[i]
        target, valid = self.targets[key]
        item.update(boundary_target=target[left:right], boundary_valid=valid[left:right])
        return item


def boundary_collate(items):
    batch = collate(items)
    batch["boundary_target"] = torch.zeros_like(batch["alive"], dtype=torch.float32)
    batch["boundary_valid"] = torch.zeros_like(batch["valid"])
    for i, item in enumerate(items):
        n = len(item["nodes"])
        batch["boundary_target"][i, :n] = torch.from_numpy(item["boundary_target"])
        batch["boundary_valid"][i, :n] = torch.from_numpy(item["boundary_valid"])
    return batch


def boundary_loss(logit, target, mask, positive_weight, kind, gamma=2.):
    error = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    weight = torch.where(target > .5, positive_weight, 1.) * mask
    if kind == "focal":
        pt = torch.where(target > .5, logit.sigmoid(), (-logit).sigmoid())
        error = error * (1 - pt).pow(gamma)
    return (error * weight).sum() / weight.sum().clamp_min(1)
