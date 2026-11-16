"""Remove only padding: invariant set encoders never observe roster indices."""
import torch
from possession.learning.data import collate
from possession.phase6.model import transition_collate


def pack(batch):
    mask=batch["node_mask"]
    count=max(1,int(mask.sum(-1).max()))
    order=torch.argsort(mask.to(torch.int8),dim=3,descending=True,stable=True)[...,:count]
    out=dict(batch)
    out["nodes"]=batch["nodes"].gather(3,order[...,None].expand(*order.shape,batch["nodes"].shape[-1]))
    out["node_mask"]=mask.gather(3,order)
    return out


def compact_collate(items):return pack(collate(items))
def compact_transition_collate(items):return pack(transition_collate(items))
