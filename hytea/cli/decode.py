import argparse
from pathlib import Path

from hytea.utils import DotDict
from hytea.bitstringdecoder import BitStringDecoder

from yaml import safe_load
import numpy as np


def decode(args: argparse.Namespace) -> None:
    """ Decodes a bitstring into a configuration. """
    with open(Path(__file__).resolve().parents[1] / 'config.yaml', 'r') as f:
        blueprint = DotDict.from_dict(safe_load(f))

    bs = BitStringDecoder(blueprint)
    config = bs.decode(np.array(args.bitstring, dtype=int))
    print(config)

    return
