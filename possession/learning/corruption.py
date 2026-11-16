"""Block missingness acts only on observed inputs. Labels are never changed."""

from dataclasses import replace
import hashlib

import numpy as np


def stable_seed(key, seed):
    return int.from_bytes(hashlib.sha256((str(seed) + ":" + key).encode()).digest()[:4], "little")


def block_mask(run, fraction, length, seed):
    if not 0 <= fraction <= 1 or length < 1:
        raise ValueError("Invalid block corruption")
    generator = np.random.default_rng(seed)
    mask = np.zeros(len(run), dtype=bool)
    for identity in np.unique(run):
        ix = np.flatnonzero(run == identity)
        # Random phase avoids alignment of block boundaries with model windows.
        offset = int(generator.integers(length))
        blocks = [ix[max(0, a):min(len(ix), a + length)] for a in range(-offset, len(ix), length)]
        covered = 0
        for j in generator.permutation(len(blocks)):
            if covered >= fraction * len(ix):
                break
            mask[blocks[j]] = True
            covered += len(blocks[j])
    return mask


def hide_ball(view, fraction, block_seconds, seed, fps=10.):
    missing = block_mask(view.run, fraction, max(1, round(block_seconds * fps)), seed)
    if fraction == 0:
        return view, missing
    nodes, global_features = view.nodes.copy(), view.global_features.copy()
    nodes[missing, ..., 2:4] = 0  # Remove all player-ball offsets, not only global XY.
    global_features[missing, :6] = 0  # XYZ, available, detected, detection_known.
    # Recompute age from remaining observations, independently per run.
    for identity in np.unique(view.run):
        last = None
        for i in np.flatnonzero(view.run == identity):
            if global_features[i, 3] > 0 and (global_features[i, 5] == 0 or global_features[i, 4] > 0):
                last = view.time[i]
            global_features[i, 6] = 1 if last is None else min(10., view.time[i] - last) / 10
    return replace(view, nodes=nodes, global_features=global_features), missing
