#!/usr/bin/env python

"""
Evolutionary Algorithm
This module contains the evolutionary algorithm class and its components.
"""

import numpy as np


class EvolutionaryAlgorithm:
    """
    Evolutionary Algorithm to recommend different neural networks.
    The bitstrings will represent the neural network architecture and its hyperparameters.

    Very simple implementation of a genetic algorithm with fixed hyperparameters to not overcomplicate things.
    - Population size: 50 or 20 idk
    - mu + lambda with mu is a fifth of the population size and lambda is the rest
    - Tournament selection with a pool size of a quarter of the population size
    - 1p crossover
    - Mutation rate: 0.1
    - 50 generations (or more if it turns out to be fast)
    """

    def __init__(self, population_size: int, candidate_size: int) -> None:
        self.population_size = population_size
        self.candidate_size = candidate_size
        self.mu_ = population_size // 5
        self.lambda_ = population_size - self.mu_
        self.pool_size = population_size // 4
        self.mutation_rate = 0.1
        self.generations = 50
    
    def run(self) -> None:
        """
        Run the evolutionary algorithm.
        """
        pass

    def initialize_population(self) -> None:
        """
        Initialize the population with random bitstrings.
        """
        self.population = np.random.randint(2, size=(self.population_size, self.candidate_size))
    
    def evaluate_population(self) -> np.ndarray:
        """
        Evaluate the population by training and testing the bitstrings.
        """
        pass

    def select(self, fitness_values: np.ndarray) -> np.ndarray:
        """
        Select the best candidates from the population.
        """
        pass

    def crossover(self, selected_candidates: np.ndarray) -> np.ndarray:
        """
        Perform crossover on the selected candidates.
        """
        pass

    def mutate(self, crossovered_candidates: np.ndarray) -> np.ndarray:
        """
        Mutate the crossovered candidates.
        """
        pass

    def replace(self, mutated_candidates: np.ndarray) -> None:
        """
        Replace the old population with the new population.
        """
        pass

    def get_best_candidate(self) -> np.ndarray:
        """
        Return the best candidate from the population.
        """
        pass

                

