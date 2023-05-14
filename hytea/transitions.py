#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data structures for storing transitions.
"""

import torch


class Trajectory:
    """ A set of transitions sampled from one episode. """

    def __init__(self, state_shape: tuple, max_length: int) -> None:
        """ Initializes a trajectory. """
        self.S = torch.empty((max_length, *state_shape), dtype=torch.float32)
        self.A = torch.empty((max_length, 1))
        self.R = torch.empty((max_length, 1), dtype=torch.float32)
        self.S_ = torch.empty((max_length, *state_shape), dtype=torch.float32)
        self.D = torch.empty((max_length, 1), dtype=torch.bool)
        self.V = torch.empty((max_length, 1))
        self.E = torch.empty((max_length, 1), dtype=torch.float32)
        self.ml = max_length
        self.l = 0

    def add(self, s: torch.Tensor, a: torch.Tensor, r: float, s_: torch.Tensor, d: bool, v: torch.Tensor, e: float) -> None:
        """ Adds a transition to the trajectory. """
        if self.full:
            raise RuntimeError('Trajectory is full.')
        self.S[self.l] = s
        self.A[self.l] = a
        self.R[self.l] = torch.tensor(r, dtype=torch.float32)
        self.S_[self.l] = s_
        self.D[self.l] = torch.tensor(d, dtype=torch.bool)
        self.V[self.l] = v
        self.E[self.l] = e
        self.l += 1

    @property
    def full(self) -> bool:
        """ Returns true if the trajectory is full. """
        return self.l == self.ml
    
    @property
    def total_reward(self) -> float:
        """ Returns the total reward of the trajectory. """
        return self.R[:self.l].sum().item()

    # this gives an error that TypeError: 'type' object is not subscriptable
    def unpack(self):
        """ Returns the trajectory as a tuple of tensors. """
        return self.S[:self.l], self.A[:self.l], self.R[:self.l], self.S_[:self.l], self.D[:self.l], self.V[:self.l], self.E[:self.l]

    def __len__(self) -> int:
        """ Returns the number of transitions in the trajectory. """
        return self.l
