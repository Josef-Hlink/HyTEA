import argparse
from pathlib import Path

from hytea.utils import DotDict
from hytea.bitstringdecoder import BitStringDecoder
from hytea.fitness import FitnessFunction
from hytea.algorithm import EvolutionaryAlgorithm
from hytea.utils.wblog import create_project_name

from yaml import safe_load


def run(args: argparse.Namespace) -> None:
    """ Runs the evolutionary algorithm. """
    args = DotDict.from_dict(vars(args))
    args.project_name = create_project_name(args)
    
    with open(Path(__file__).resolve().parents[1] / 'config.yaml', 'r') as f:
        config = DotDict.from_dict(safe_load(f))

    bs = BitStringDecoder(config)
    ff = FitnessFunction(args, bs)
    ea = EvolutionaryAlgorithm(args, ff)
    ea.run()
    
    return
