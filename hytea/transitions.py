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
        self.A = torch.empty((max_length, 1), dtype=torch.int8)
        self.R = torch.empty((max_length, 1), dtype=torch.float32)
        self.S_ = torch.empty((max_length, *state_shape), dtype=torch.float32)
        self.D = torch.empty((max_length, 1), dtype=torch.bool)
        self.l = 0

    def add(self, s: torch.Tensor, a: int, r: float, s_: torch.Tensor, d: bool) -> None:
        """ Adds a transition to the trajectory. """
        if self.l == len(self):
            raise RuntimeError('Trajectory is full.')
        self.S[self.l] = s
        self.A[self.l] = torch.tensor(a, dtype=torch.int8)
        self.R[self.l] = torch.tensor(r, dtype=torch.float32)
        self.S_[self.l] = s_
        self.D[self.l] = torch.tensor(d, dtype=torch.bool)
        self.l += 1

    def unpack(self) -> tuple:
        """ Returns the trajectory as a tuple of tensors. """
        return self.S[:self.l], self.A[:self.l], self.R[:self.l], self.S_[:self.l], self.D[:self.l]

    def __len__(self) -> int:
        """ Returns the number of transitions in the trajectory. """
        return self.l
