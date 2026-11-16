"""Shared probability decoder contract. Columns always H, A, D."""

from typing import Protocol

import numpy as np


def validate_probabilities(probabilities):
    values = np.asarray(probabilities)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("Expected finite [T,3] H/A/D probabilities")
    if np.any(values < 0) or not np.allclose(values.sum(-1), 1, atol=1e-5):
        raise ValueError("Invalid probability simplex")
    return values


class Decoder(Protocol):
    def decode(self, probabilities: np.ndarray, timestamps: np.ndarray) -> np.ndarray: ...


class ArgmaxDecoder:
    def decode(self, probabilities, timestamps):
        values = validate_probabilities(probabilities)
        if len(timestamps) != len(values) or np.any(np.diff(timestamps) <= 0):
            raise ValueError("Decode one strictly monotone contiguous run at a time")
        return values.argmax(-1).astype(np.int8)


def decoder_from_config(name="argmax"):
    if name != "argmax":
        raise NotImplementedError("HMM/HSMM are reserved interfaces, not implemented in phase 1")
    return ArgmaxDecoder()
