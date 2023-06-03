#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name = 'hytea',
    version = '0.1.0',
    description = 'Hyperparameter Tuning using Evolutionary Algorithms',
    author = 'Josef Hamelink & Shreyansh Sharma',
    packages = find_packages(),
    entry_points = {
        'console_scripts': [
            'hytea = hytea.cli:cli',
            'hytea-decode = hytea.bitstring:cli',
            'hytea-test = hytea.test_run:cli',
            'hytea-quickie = hytea.quickie:cli'
        ],
    },
    install_requires = [
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'jupyter',
        'ipykernel',
        'seaborn',
        'gymnasium[box2d]',
        'torch',
        'wandb',
        'tqdm',
        'randomname'
    ],
)
