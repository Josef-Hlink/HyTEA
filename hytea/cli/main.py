#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from hytea.cli import run, add_run_args, decode, add_decode_args, test, add_test_args


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


if __name__ == '__main__':
    main()
