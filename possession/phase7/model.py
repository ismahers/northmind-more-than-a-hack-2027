"""Small masked SAB+PMA spatial encoder, with the unchanged E0 temporal model.

SAB/PMA follow Lee et al., ICML 2019 (https://proceedings.mlr.press/v97/lee19d.html).
Implementation is independent; no positional/player embeddings. Team membership
is retained, and each team receives two shared pooling queries after cross-team
interaction. All-empty sets and padded query tokens are explicitly masked.
"""

import torch
from torch import nn
from torch.nn import functional as F
from possession.learning.model import E0


class AttentionBlock(nn.Module):
    def __init__(self, width=32, heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2*width), nn.ReLU(), nn.Linear(2*width, width))

    def forward(self, query, keys, present):
        # MHA otherwise produces NaN for a row whose every key is masked.
        empty = ~present.any(-1)
        safe = present.clone(); safe[empty, 0] = True
        keys = keys * present.unsqueeze(-1)
        result = self.attention(query, keys, keys, key_padding_mask=~safe, need_weights=False)[0]
        x = self.norm1(query + result)
        return self.norm2(x + self.ff(x))


class SetSpatial(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.embedding = nn.Linear(6, width)
        self.team_embedding = nn.Parameter(torch.empty(2, width))
        self.sab = AttentionBlock(width)
        self.pma = AttentionBlock(width)
        self.seeds = nn.Parameter(torch.empty(1, 2, width))
        nn.init.normal_(self.team_embedding, std=.02)
        nn.init.xavier_uniform_(self.seeds)

    def _frames(self, nodes, mask):
        n, _, p, _ = nodes.shape
        h = self.embedding(nodes) + self.team_embedding[None, :, None, :]
        h = h.flatten(1, 2); present = mask.flatten(1, 2)
        h = self.sab(h, h, present) * present[..., None]
        # Pool team members after they have seen players from BOTH teams.
        h = h.reshape(n*2, p, -1); present = mask.reshape(n*2, p)
        pooled = self.pma(self.seeds.expand(n*2, -1, -1), h, present)
        pooled = pooled * present.any(-1)[:, None, None]
        return pooled.reshape(n, 128)

    def forward(self, nodes, mask):
        b, t, _, p, _ = nodes.shape
        flat, keep = nodes.reshape(b*t, 2, p, 6), mask.reshape(b*t, 2, p)
        # Bounded attention temporaries; no attention across time/windows.
        return torch.cat([self._frames(flat[a:a+256], keep[a:a+256])
                          for a in range(0, len(flat), 256)]).reshape(b, t, 128)


class SpatialE0(E0):
    def __init__(self, spatial="harmonized", auxiliary=False):
        super().__init__()
        self.spatial, self.auxiliary, self.directions = spatial, auxiliary, 1
        if spatial == "set_transformer":
            # Common TCN/heads and stochastic training stream match DeepSets.
            state = torch.get_rng_state()
            self.encoder = SetSpatial()
            torch.set_rng_state(state)
        if auxiliary:
            # Match the original TransitionE0 binary-head initialization exactly.
            self.boundary_head = nn.Conv1d(64, 1, 1)
            self.boundary_head = nn.Conv1d(64, 1, 1)

    def forward(self, nodes, node_mask, global_features, valid):
        if self.spatial == "set_transformer":
            pooled = self.encoder(nodes, node_mask)
        else:
            h = self.encoder(nodes); w = node_mask.unsqueeze(-1)
            mean = (h*w).sum(-2)/w.sum(-2).clamp_min(1)
            maximum = h.masked_fill(~w, -torch.inf).amax(-2)
            maximum = torch.where(node_mask.any(-1, keepdim=True), maximum, torch.zeros_like(maximum))
            pooled = torch.cat((mean, maximum), -1).flatten(-2)
        mask = valid.unsqueeze(1).to(nodes.dtype)
        x = self.input(torch.cat((pooled, global_features), -1).transpose(1, 2))*mask
        for block in self.blocks:
            x = block(x, mask)
        alive, team = self.alive_head(x).squeeze(1), self.team_head(x).squeeze(1)
        a, h = alive.sigmoid(), team.sigmoid()
        out = {"alive_logit": alive, "team_logit": team,
               "probabilities": torch.stack((a*h, a*(1-h), 1-a), -1)}
        if self.auxiliary:
            out["boundary_logit"] = self.boundary_head(x).transpose(1, 2)
        return out
