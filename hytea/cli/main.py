#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from hytea.cli import run, decode, test


def main() -> None:
    """ Entry point for the command line interface. """

    # top-level parser
    parser = argparse.ArgumentParser(
        description = 'Hyperparameter Tuning using Evolutionary Algorithms',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='cmd', help='Choose one of the following subcommands:')
    subparser_run = subparsers.add_parser('run', help='Run the evolutionary algorithm.')
    subparser_decode = subparsers.add_parser('decode', help='Decode a bitstring.')
    subparser_test = subparsers.add_parser('test', help='Run a single agent.')

    subparser_run = add_run_args(subparser_run)
    subparser_decode = add_decode_args(subparser_decode)
    subparser_test = add_test_args(subparser_test)

    args = parser.parse_args()

    if args.cmd == 'run': run(args)
    elif args.cmd == 'decode': decode(args)
    elif args.cmd == 'test': test(args)
    else: raise ValueError(f'Unknown command: {args.cmd}')
    
    return

def add_run_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """ Add arguments to the "run" subparser. """
    parser.add_argument('--env', dest='env_name', type=str,
        default='LunarLander-v2', help='The environment to train the agent in.'
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

def add_decode_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """ Add arguments to the "decode" subparser. """
    parser.add_argument('bitstring', nargs='+', help='The bitstring to decode.')
    return parser

def add_test_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--env_name', type=str,
        default='LunarLander-v2', help='The environment to use.'
    )
    parser.add_argument('--num_train_episodes', type=int,
        default=1000, help='The number of episodes to train for.'
    )
    parser.add_argument('--num_test_episodes', type=int,
        default=100, help='The number of episodes to test for.'
    )
    parser.add_argument('--num_runs', type=int,
        default=1, help='The number of individual runs to average over.'
    )
    return parser


if __name__ == '__main__':
    main()
