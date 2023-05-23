#!/usr/bin/env python3

"""
Encoder and decoder for the bitstrings.
"""

import numpy as np
from pathlib import Path

from yaml import safe_load


class BitStringDecoder():
    """	
    Encoder for the bitstrings.	
    """
    
    def __init__(self, population_size: int, config_path: str) -> None:
        self.population_size = population_size
        self.bitstring_config = self._load_config(config_path)
        self.candidate_size = np.sum([v['bits'] for v in self.bitstring_config.values()])
        return
    
    def _load_config(self, config_path: str) -> dict:
        """
        Load the bitstring config from yaml.
        """
        if not Path(config_path).exists():
            raise FileNotFoundError(f'BitString Configuration file {config_path} does not exist.')

        with open(config_path, 'r') as f:
            config = safe_load(f)
        
        return config
    
    def decode(self, bitstring: np.ndarray) -> np.ndarray:
        """
        decode a single bitstring into a dictionary.
        
        Arguments:
        - bitstring: The bitstring to decode.
        """
        decoded = {}
        bitstring = [str(b) for b in bitstring]
        bitstring = "".join(bitstring)
        
        for k,v in self.bitstring_config.items():
            b = v['bits']
            if "key" in v:
                if v["key"] not in decoded:
                    decoded[v["key"]] = {}
                decoded[v["key"]][k] = v["values"][int("".join(bitstring[:b]), 2)]
            else:
                decoded[k] = v["values"][int("".join(bitstring[:b]), 2)]
            bitstring = bitstring[b:]
        return decoded