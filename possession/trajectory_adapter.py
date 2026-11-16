"""Player-aligned temporal evidence BEFORE spatial pooling; no identity features."""
import copy
import numpy as np
import torch
from torch import nn

from possession.learning.data import Windows, collate
from possession.learning.training import forward

OFFSETS = (-10, -5, -1, 0, 1, 5, 10)


def track_collate(items):
    """Compact once per window/team, never independently per frame."""
    batch = collate(items)
    mask = batch["node_mask"]
    present = mask.any(1)
    count = max(1, int(present.sum(-1).max()))
    order = torch.argsort(present.to(torch.int8), dim=-1, descending=True, stable=True)[..., :count]
    index = order[:, None].expand(-1, mask.shape[1], -1, -1)
    batch["nodes"] = batch["nodes"].gather(3, index[..., None].expand(*index.shape, 6))
    batch["node_mask"] = mask.gather(3, index)
    return batch


def track_features(nodes, mask, temporal):
    """Seven aligned taps plus availability; never bridge a missing-player gap."""
    taps = []
    for offset in OFFSETS:
        if not temporal or offset == 0:
            value, visible = nodes, mask
        else:
            value = torch.zeros_like(nodes)
            visible = torch.zeros_like(mask)
            d, n = abs(offset), nodes.shape[1]
            if n > d:
                dst = slice(0, n-d) if offset > 0 else slice(d, n)
                src = slice(d, n) if offset > 0 else slice(0, n-d)
                continuous = torch.ones_like(mask[:, dst])
                # All intermediate observations must belong to the same visible track.
                for step in range(d+1):
                    continuous &= mask[:, step:n-d+step]
                value[:, dst] = nodes[:, src]
                visible[:, dst] = continuous
        taps.extend((value * visible[..., None], visible[..., None].to(nodes.dtype)))
    return torch.cat(taps, -1)


class TrackSpatial(nn.Module):
    def __init__(self, original, temporal):
        super().__init__()
        self.original = copy.deepcopy(original)
        self.temporal = temporal
        self.adapter = nn.Sequential(nn.Linear(49, 32), nn.ReLU(), nn.Linear(32, 32))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, nodes, mask):
        b, t, _, p, _ = nodes.shape
        residual = self.adapter(track_features(nodes, mask, self.temporal)).reshape(b*t, 2, p, 32)
        nodes, mask = nodes.reshape(b*t, 2, p, 6), mask.reshape(b*t, 2, p)
        outputs = []
        for a in range(0, b*t, 256):
            x, keep = nodes[a:a+256], mask[a:a+256]
            n = len(x)
            h = self.original.embedding(x) + self.original.team_embedding[None, :, None] + residual[a:a+256]
            h, present = h.flatten(1, 2), keep.flatten(1, 2)
            h = self.original.sab(h, h, present) * present[..., None]
            h, present = h.reshape(n*2, p, 32), keep.reshape(n*2, p)
            pooled = self.original.pma(self.original.seeds.expand(n*2, -1, -1), h, present)
            outputs.append((pooled * present.any(-1)[:, None, None]).reshape(n, 128))
        return torch.cat(outputs).reshape(b, t, 128)


def adapted(parent, temporal):
    model = copy.deepcopy(parent)
    model.encoder = TrackSpatial(parent.encoder, temporal)
    model.context = 25  # 15 TCN + 10 trajectory, same windows for matched control.
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("encoder.adapter."))
    return model


def predict(model, view):
    """Inference consumes geometry/masks only; labels in Windows are never forwarded."""
    data = Windows({"public": view}, 256, 25)
    probabilities = np.zeros((len(view.time), 3), np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data), 8):
            items = [data[i] for i in range(start, min(start+8, len(data)))]
            out = forward(model, track_collate(items))["probabilities"].numpy()
            for item, value in zip(items, out):
                a, b, left = item["a"], item["b"], item["left"]
                probabilities[a:b] = value[a-left:b-left]
    return probabilities
