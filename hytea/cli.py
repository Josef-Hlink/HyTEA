#!/usr/bin/env python

import argparse
from pathlib import Path
from yaml import safe_load

from hytea.utils import DotDict
from hytea.fitness import FitnessFunction
from hytea.bitstring import BitStringDecoder
from hytea.algorithm import EvolutionaryAlgorithm


def cli() -> None:
    
    config_path = Path(__file__).resolve().parents[0] / 'config.yaml'
    with open(config_path, 'r') as f:
        config = DotDict.from_dict(safe_load(f))

    bs = BitStringDecoder(config)
    ff = FitnessFunction(bs)
    ea = EvolutionaryAlgorithm(50, bs.get_candidate_size(), ff)
    ea.run()

    return


if __name__ == '__main__':
    cli()
