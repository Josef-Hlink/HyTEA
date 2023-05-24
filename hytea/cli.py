#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from pathlib import Path
from yaml import safe_load

from hytea.utils import DotDict
from hytea.fitness import FitnessFunction
from hytea.bitstring import BitStringDecoder
from hytea.algorithm import EvolutionaryAlgorithm


def cli() -> None:

    args = parse_args()
    
    with open(Path(__file__).resolve().parents[0] / 'config.yaml', 'r') as f:
        config = DotDict.from_dict(safe_load(f))

    bs = BitStringDecoder(config)
    ff = FitnessFunction(bs)
    ea = EvolutionaryAlgorithm(args.num_generations, args.population_size, ff)
    ea.run()

    return

def parse_args() -> Namespace:
    """ Parse the command line arguments. """
    parser = ArgumentParser(
        description = 'Hyperparameter Tuning Evolutionary Algorithm',
        formatter_class = ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--env', dest='env_name', type=str, default='LunarLander-v2', help='The environment to train the agent in.')
    parser.add_argument('--train', dest='num_train_episodes', type=int, default=1000, help='The number of episodes to train the agent for.')
    parser.add_argument('--test', dest='num_test_episodes', type=int, default=100, help='The number of episodes to test the agent for.')
    parser.add_argument('--runs', dest='num_runs', type=int, default=3, help='The number of runs to average the reward over.')
    parser.add_argument('--pop', dest='population_size', type=int, default=10, help='The size of an EA population.')
    parser.add_argument('--gen', dest='num_generations', type=int, default=10, help='The number of generations to run the EA for.')
    return parser.parse_args()


if __name__ == '__main__':
    cli()
