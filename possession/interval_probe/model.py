"""Conditional team, positive L/R offsets and localization quality; D stays frozen."""
import torch
from torch import nn
from torch.nn import functional as F


class IntervalHead(nn.Module):
    def __init__(self,team_weight,team_bias):
        super().__init__()
        self.register_buffer("original_team_weight",team_weight.detach().reshape(64).clone())
        self.register_buffer("original_team_bias",team_bias.detach().reshape(()).clone())
        self.hidden=nn.Linear(64,64);self.output=nn.Linear(64,7)
        # Initial conditional-team estimate equals the teacher, avoiding a random reset.
        with torch.no_grad():self.output.weight[0]=0;self.output.bias[0]=0

    def forward(self,x):
        y=self.output(F.relu(self.hidden(x)))
        team=x@self.original_team_weight+self.original_team_bias+y[:,0]
        distances=3*F.softplus(y[:,1:5].reshape(-1,2,2))
        return team,distances,y[:,5:7]


def interval_loss(output,target,kind="iou"):
    team,distance,quality=output
    label=target["label"];live=label<2;which=label.clamp_max(1)
    weight=target["condition_weight"]
    cls=F.binary_cross_entropy_with_logits(team,(label==0).float(),reduction="none")
    cls=(cls*weight*live).sum()/(weight*live).sum().clamp_min(1)
    rows=torch.arange(len(label));pred=distance[rows,which];truth=target["distance"]
    intersection=torch.minimum(pred,truth).sum(1)
    union=torch.maximum(pred,truth).sum(1).clamp_min(1e-6)
    error=1-intersection/union
    if kind=="diou":
        difference=((pred[:,1]-pred[:,0])-(truth[:,1]-truth[:,0]))/2
        error=error+(difference/union).square()
    elif kind!="iou":raise ValueError("Unknown interval loss")
    rw=target["regression_weight"]*weight*target["regression_valid"]
    regression=((error+.25*F.smooth_l1_loss(torch.log1p(pred),torch.log1p(truth),reduction="none").mean(1))*rw).sum()/rw.sum().clamp_min(1)
    qw=weight*target["regression_valid"]
    # Centerness is zero if the only short-spell anchor is its left boundary.
    # Detached IoU quality avoids suppressing that valid interval by definition.
    quality_target=(intersection/union).detach()
    quality_error=F.binary_cross_entropy_with_logits(quality[rows,which],quality_target,reduction="none")
    quality_error=(quality_error*qw).sum()/qw.sum().clamp_min(1)
    return cls+regression+quality_error,{"conditional_team":float(cls.detach()),"regression":float(regression.detach()),"localization_quality":float(quality_error.detach())}
