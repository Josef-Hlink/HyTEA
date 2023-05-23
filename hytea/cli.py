#!/usr/bin/env python

import argparse
from pathlib import Path
from yaml import safe_load

from hytea.utils import DotDict
from hytea.main import main
from hytea.algorithm import main as ea_main


def cli() -> None:
    parser = argparse.ArgumentParser(
        description = 'Hyperparameter Tuning using Evolutionary Algorithms',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('config', type=str, help='Path to the configuration file.')

    args = parser.parse_args()

    if not Path(args.config).exists():
        raise FileNotFoundError(f'Configuration file {args.config} does not exist.')

    with open(args.config, 'r') as f:
        config = DotDict.from_dict(safe_load(f))

    main(config=config, caller='cli')
    
def ea_cli() -> None:
    parser = argparse.ArgumentParser(
        description = 'Hyperparameter Tuning using Evolutionary Algorithms',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('config', type=str, help='Path to the configuration file.')

    args = parser.parse_args()

    if not Path(args.config).exists():
        raise FileNotFoundError(f'Configuration file {args.config} does not exist.')

    with open(args.config, 'r') as f:
        config = DotDict.from_dict(safe_load(f))

    ea_main(config=config, caller='cli')

if __name__ == '__main__':
    cli()
