from hytea.utils import DotDict

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

    @property
    def separator(self) -> str:
        """ Separator for the bitstring. """
        string = '|'
        for scope in self.blueprint.keys():
            for k, v in self.blueprint[scope].items():
                num_spaces = 1 + (v.bits - 1) * 2
                string += ' ' * num_spaces + '|'
        return string

