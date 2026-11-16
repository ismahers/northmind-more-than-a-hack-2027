"""A geometry contract with identical semantics across providers.

Unused legacy slots stay zero to allow exact warm starts and input ablations.
These slots are NOT unknown heights or detection observations consumed by a model.
Only coordinates, explicit availability and elapsed time are active features.
"""

from dataclasses import replace
import numpy as np


def availability_age(view):
    age = np.ones(len(view.time), np.float32)
    for run in np.unique(view.run):
        ix = np.flatnonzero(view.run == run)
        observed = view.global_features[ix, 3] > .5
        stamp = np.where(observed, view.time[ix], -np.inf)
        last = np.maximum.accumulate(stamp)
        age[ix] = np.minimum(10., view.time[ix] - last) / 10
    return age


def intervene(view, condition):
    nodes, g = view.nodes.copy(), view.global_features.copy()
    if condition in ("player_flags", "all_flags", "harmonized"):
        nodes[..., 4:6] = 0
    if condition in ("ball_flags", "all_flags", "harmonized"):
        g[:, 4:6] = 0
    if condition == "known_only":
        nodes[..., 5] = 0; g[:, 5] = 0
    if condition == "detected_only":
        nodes[..., 4] = 0; g[:, 4] = 0
    if condition in ("age", "all_flags", "harmonized"):
        g[:, 6] = availability_age(view)
    if condition == "harmonized":
        # ATD does not promise ball height. Never treat another provider's speed as Z.
        g[:, 2] = 0
    if condition not in ("original", "player_flags", "ball_flags", "known_only",
                         "detected_only", "age", "all_flags", "harmonized"):
        raise ValueError("Unknown public feature intervention")
    return replace(view, nodes=nodes, global_features=g)


def harmonize(view):
    return intervene(view, "harmonized")
