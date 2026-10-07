"""Seeded 3-state, 2-action MDP used by the tests and the demo.

The rewards make action 1 strictly optimal. The perturbation moves logits of
that action so the occupancy changes while the optimal basis stays put. Neither
the rewards nor the perturbation come from the paper's benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ToyMDP:
    logits: np.ndarray
    reward: np.ndarray
    gamma: np.ndarray
    beta: float

    @property
    def n_states(self):
        return int(self.logits.shape[0])

    @property
    def n_actions(self):
        return int(self.logits.shape[1])


def make_toy_mdp(seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(scale=0.4, size=(3, 2, 3))
    reward = np.array(
        [
            [0.0, 1.5],
            [0.1, 1.0],
            [-0.2, 1.2],
        ],
        dtype=np.float64,
    )
    gamma = np.array([0.5, 0.3, 0.2], dtype=np.float64)
    return ToyMDP(logits=logits, reward=reward, gamma=gamma, beta=0.9)


def same_basis_perturbation(logits):
    """Perturbation that changes y on the seed-0 toy without changing the basis."""
    out = np.array(logits, dtype=np.float64, copy=True)
    out[0, 1, 0] += 0.8
    out[1, 1, 1] -= 0.5
    return out
