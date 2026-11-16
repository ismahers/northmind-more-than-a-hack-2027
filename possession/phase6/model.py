"""Same E0 trunk and independent alive/team heads; one or two event scores."""

import numpy as np
import torch
from torch import nn

from possession.external_data.audit import segments, joint_labels
from possession.phase5.model import BoundaryE0
from possession.learning.data import Windows, collate


class TransitionE0(BoundaryE0):
    def __init__(self, directions=2, **kwargs):
        super().__init__(**kwargs)
        self.directions = directions
        self.boundary_head = nn.Conv1d(self.team_head.in_channels, directions, 1)

    def forward(self, nodes, node_mask, global_features, valid):
        result = super().forward(nodes, node_mask, global_features, valid)
        q = result["boundary_logit"]
        result["boundary_logit"] = q.unsqueeze(-1) if self.directions == 1 else q.transpose(1, 2)
        return result


def transition_targets(match, view, directions=2, tolerance=.12):
    y = joint_labels(match)
    target = np.zeros((len(view.time), directions), np.float32)
    valid = np.zeros(len(view.time), bool)
    for run, (a, b) in enumerate(segments(match)):
        ix = np.flatnonzero(view.run == run)
        time = view.time[ix]
        native = y[a:b]
        changes = np.flatnonzero((native[:-1] >= 0) & (native[:-1] < 2) &
                                 (native[1:] >= 0) & (native[1:] < 2) & (native[:-1] != native[1:])) + 1
        for j in changes:
            event = match.time[a + j]
            lo, hi = np.searchsorted(time, [event-tolerance, event+tolerance])
            direction = int(native[j-1]) if directions == 2 else 0
            target[ix[lo:hi], direction] = 1
        known = np.ones(len(ix), bool)
        for bad in match.time[a:b][native < 0]:
            known &= abs(time-bad) > tolerance
        live = (view.alive[ix] == 1) & (view.team[ix] >= 0)
        target[ix] *= live[:, None]
        valid[ix] = known & ((view.alive[ix] == 0) | live)
    return target, valid


class TransitionWindows(Windows):
    def __init__(self, views, targets, context=15):
        super().__init__(views, 256, context)
        self.targets = targets

    def __getitem__(self, i):
        item = super().__getitem__(i)
        key, left, right, _, _ = self.windows[i]
        y, valid = self.targets[key]
        item.update(transition_target=y[left:right], transition_valid=valid[left:right])
        return item


def transition_collate(items):
    result = collate(items)
    shape = (*result["valid"].shape, items[0]["transition_target"].shape[1])
    result["transition_target"] = torch.zeros(shape)
    result["transition_valid"] = torch.zeros_like(result["valid"])
    for i, item in enumerate(items):
        n = len(item["nodes"])
        result["transition_target"][i, :n] = torch.from_numpy(item["transition_target"])
        result["transition_valid"][i, :n] = torch.from_numpy(item["transition_valid"])
    return result
