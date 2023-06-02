#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

from hytea.utils import DotDict

from yaml import safe_load
import numpy as np


class BitStringDecoder():
    """	Encoder for the bitstrings.	"""
    
    def __init__(self, config: DotDict) -> None:
        self.blueprint = config
        return
    
    def get_candidate_size(self) -> int:
        """ Calculates the size of the candidate bitstring. """
        return sum([v.bits for scope in self.blueprint.keys() for v in self.blueprint[scope].values()])
    
    def decode(self, bitstring: np.ndarray) -> DotDict:
        """ Decodes a bitstring into a configuration.
        
        ### Args:
        `np.ndarray` bitstring: The bitstring to decode.
        """
        bitstring: str = ''.join([str(b) for b in bitstring])
        config = DotDict(agent = DotDict(), network = DotDict(), optimizer = DotDict())

        for scope in config.keys():
            for k, v in self.blueprint[scope].items():
                b: int = v.bits
                i: int = int(bitstring[:b], 2)
                config[scope][k] = v.vals[i]
                bitstring = bitstring[b:]

        return config


def cli() -> None:
    parser = argparse.ArgumentParser(description='Bitstring decoder.')
    parser.add_argument('bitstring', nargs='+', help='The bitstring to decode.')
    args = parser.parse_args()

    with open(Path(__file__).resolve().parents[0] / 'config.yaml', 'r') as f:
        blueprint = DotDict.from_dict(safe_load(f))

    bs = BitStringDecoder(blueprint)
    config = bs.decode(np.array(args.bitstring, dtype=int))
    print(config)
    return


if __name__ == '__main__':
    print(cli())
