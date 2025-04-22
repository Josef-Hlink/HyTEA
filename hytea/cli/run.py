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

    assert args.population_size % 2 == 0, 'Population size must be even.'

    args.project_name = create_project_name(args)
    
    with open(Path(__file__).resolve().parents[1] / 'config.yaml', 'r') as f:
        config = DotDict.from_dict(safe_load(f))

    bs = BitStringDecoder(config)
    ff = FitnessFunction(args, bs)
    ea = EvolutionaryAlgorithm(args, ff)
    ea.run()
    
    return

def add_run_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """ Add arguments to the "run" subparser. """
    parser.add_argument('--env', dest='env_name', type=str,
        default='LunarLander-v3', help='The environment to train the agent in.'
    )
    parser.add_argument('--gen', dest='num_generations', type=int,
        default=5, help='The number of generations to run the EA for.'
    )
    parser.add_argument('--pop', dest='population_size', type=int,
        default=4, help='The size of an EA population.'
    )
    parser.add_argument('--train', dest='num_train_episodes', type=int,
        default=1000, help='The number of episodes to train the agent for.'
    )
    parser.add_argument('--test', dest='num_test_episodes', type=int,
        default=100, help='The number of episodes to test the agent for.'
    )
    parser.add_argument('--runs', dest='num_runs', type=int,
        default=3, help='The number of runs to average the reward over.'
    )
    parser.add_argument('--wt', dest='wandb_team', type=str,
        default='hytea', help='The name of the wandb team.'
    )
    parser.add_argument('-W', dest='use_wandb', action='store_true', help='Use wandb for logging.')
    return parser
