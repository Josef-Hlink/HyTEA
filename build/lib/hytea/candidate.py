#!/usr/bin env python3

import numpy as np


class Candidate:

    def __init__(self, genome: np.ndarray):
        assert genome.shape == (18,)
        self.genome = genome

    @property
    def nHidden(self) -> int:
        """ 00 -> 1, 01 -> 2, 10 -> 3, 11 -> 4 """
        sec = self.genome[0:2]
        return int(sec[0] + 2 * sec[1]) + 1
    
    @property
    def dropOutRate(self) -> float:
        """ 00 -> 0.0, 01 -> 0.1, 10 -> 0.2, 11 -> 0.3 """
        sec = self.genome[2:4]
        return float(sec[0] + 2 * sec[1]) / 10.0
    
    @property
    def nNeurons(self) -> int:
        """ 000 -> 2^1, 001 -> 2^2, 010 -> 2^3, ..., 111 -> 2^8 """
        sec = self.genome[4:7]
        return 2 ** (int(4 * sec[0] + 2 * sec[1] + sec[2]) + 1)
    