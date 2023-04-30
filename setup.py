#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name = 'hytea',
    version = '0.0.1',
    description = 'Hyperparameter Tuning using Evolutionary Algorithms',
    author = 'Josef Hamelink & Shreyansh Sharma',
    packages = find_packages(),
    entry_points = {
        'console_scripts': [
            'hytea = hytea.cli:main',
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
        'gym',
        'torch',
        'wandb',
    ],
)
