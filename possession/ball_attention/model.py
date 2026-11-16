"""Change only PMA queries; preserve SAB, packing, temporal blocks and heads."""
import torch
from torch import nn
from possession.phase7.model import SpatialE0, SetSpatial


class QuerySpatial(SetSpatial):
    def __init__(self, arm):
        super().__init__()
        if arm not in ("original", "ball", "capacity"):
            raise ValueError("Unknown query arm")
        self.arm = arm
        if arm != "original":
            self.query_mlp = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 64))
            nn.init.zeros_(self.query_mlp[-1].weight)
            nn.init.zeros_(self.query_mlp[-1].bias)

    def queries(self, global_features):
        n = len(global_features)
        seeds = self.seeds.expand(n, -1, -1)
        if self.arm == "original":
            return seeds
        if self.arm == "ball":
            available = global_features[:, 3] > .5
            xy = torch.where(available[:, None], global_features[:, :2], 0.)
            residual = self.query_mlp(xy).reshape(n, 2, 32)
            residual = torch.where(available[:, None, None], residual, 0.)
        else:
            residual = self.query_mlp(torch.ones_like(global_features[:, :2])).reshape(n, 2, 32)
        return seeds + residual

    def _conditioned_frames(self, nodes, mask, g):
        n, _, p, _ = nodes.shape
        h = self.embedding(nodes) + self.team_embedding[None, :, None, :]
        h = h.flatten(1, 2); present = mask.flatten(1, 2)
        h = self.sab(h, h, present) * present[..., None]
        h = h.reshape(n*2, p, -1); present = mask.reshape(n*2, p)
        # One ball-derived query pair, shared by the two team pools as before.
        q = self.queries(g)[:, None].expand(-1, 2, -1, -1).reshape(n*2, 2, 32)
        pooled = self.pma(q, h, present) * present.any(-1)[:, None, None]
        return pooled.reshape(n, 128)

    def forward(self, nodes, mask, global_features):
        if self.arm == "original":
            return super().forward(nodes, mask)
        b, t, _, p, _ = nodes.shape
        flat, keep = nodes.reshape(b*t, 2, p, 6), mask.reshape(b*t, 2, p)
        g = global_features.reshape(b*t, -1)
        return torch.cat([self._conditioned_frames(flat[a:a+256], keep[a:a+256], g[a:a+256])
                          for a in range(0, len(flat), 256)]).reshape(b, t, 128)


class QueryE0(SpatialE0):
    def __init__(self, arm="original", auxiliary=False):
        super().__init__("set_transformer", auxiliary)
        state = torch.get_rng_state()
        old = self.encoder.state_dict()
        self.encoder = QuerySpatial(arm)
        missing, extra = self.encoder.load_state_dict(old, strict=False)
        assert not extra and all(name.startswith("query_mlp.") for name in missing)
        # New query initialization must not change dropout/head/sampler streams.
        torch.set_rng_state(state)

    def forward(self, nodes, node_mask, global_features, valid):
        pooled = self.encoder(nodes, node_mask, global_features)
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
