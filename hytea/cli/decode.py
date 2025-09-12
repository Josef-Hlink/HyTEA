import argparse
from pathlib import Path

import numpy as np
from yaml import safe_load

from hytea.bitstringdecoder import BitStringDecoder
from hytea.utils import DotDict


def decode(args: argparse.Namespace) -> None:
    """Decodes a bitstring into a configuration."""
    with open(Path(__file__).resolve().parents[1] / 'config.yaml', 'r') as f:
        blueprint = DotDict.from_dict(safe_load(f))

    bs = BitStringDecoder(blueprint)
    config = bs.decode(np.array(args.bitstring, dtype=int))
    print(config)

    return


def add_decode_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add arguments to the "decode" subparser."""
    parser.add_argument('bitstring', nargs='+', help='The bitstring to decode.')
    return parser
