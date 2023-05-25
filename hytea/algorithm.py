#!/usr/bin/env python

"""
Evolutionary Algorithm
This module contains the evolutionary algorithm class and its components.
"""

from multiprocessing import Pool

from hytea.fitness import FitnessFunction

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

    def __init__(self,
            num_generations: int,
            population_size: int,
            fitness_function: FitnessFunction
    ) -> None:
        """ Initialize the evolutionary algorithm.

        ### Args:
        `int` num_generations: The number of generations to run the algorithm for.
        `int` population_size: The size of the population.
        `FitnessFunction` fitness_function: The fitness function to evaluate the candidates with.
            This will be used to evaluate the population, and to get the size of a candidate bitstring.
        """

        self.num_generations = num_generations
        self.population_size = population_size
        
        self.mu_ = population_size // 5
        self.lambda_ = population_size - self.mu_
        self.pool_size = population_size // 4
        self.mutation_rate = 0.1

        self.evaluate = fitness_function.evaluate
        self.candidate_size = fitness_function.decoder.get_candidate_size()

        return
    
    def run(self) -> None:
        """ Main loop of the evolutionary algorithm. """
        self.initialize_population()

        for i in range(self.num_generations):
            fitness_values = self.evaluate_population()
            print(f"Generation {i}: {np.mean(fitness_values)}")
            selected_candidates = self.select(fitness_values)
            crossovered_candidates = self.crossover(selected_candidates)
            mutated_candidates = self.mutate(crossovered_candidates)
            self.replace(mutated_candidates)
        
        best_candidate = self.get_best_candidate()
        print(best_candidate)

    def initialize_population(self) -> None:
        """
        Initialize the population with random bitstrings.
        """
        self.population = np.random.randint(2, size=(self.population_size, self.candidate_size), dtype=np.uint8)
    
    def evaluate_population(self) -> np.ndarray:
        """ Evaluate the population by training and testing the bitstrings. """
        with Pool() as pool:
            return np.array(pool.map(self.evaluate, self.population))

    def select(self, fitness_values: np.ndarray) -> np.ndarray:
        """
        Select the best candidates from the population using tournament selection.
        """
        selected_candidates = np.empty((self.mu_, self.candidate_size))

        for i in range(self.mu_):
            tournament_indices = np.random.choice(len(self.population), self.pool_size, replace=False)
            tournament = self.population[tournament_indices]
            best_individual_index = np.argmax(fitness_values[tournament_indices])
            best_individual = tournament[best_individual_index]
            selected_candidates[i] = best_individual

        return selected_candidates


    def crossover(self, selected_candidates: np.ndarray) -> np.ndarray:
        """
        Perform crossover on the selected candidates.
        """
        crossovered_candidates = np.empty((self.lambda_, selected_candidates.shape[1]))

        for i in range(0, self.lambda_, 2):
            parent1_index = i % self.mu_
            parent2_index = (i + 1) % self.mu_

            parent1 = selected_candidates[parent1_index]
            parent2 = selected_candidates[parent2_index]

            crossover_point = np.random.randint(1, len(parent1) - 1)
            offspring1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            offspring2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))

            crossovered_candidates[i] = offspring1
            crossovered_candidates[i + 1] = offspring2
        
        return np.concatenate((selected_candidates, crossovered_candidates))

    def mutate(self, crossovered_candidates: np.ndarray) -> np.ndarray:
        """
        Mutate the crossovered candidates.
        """
        mask = np.random.rand(*crossovered_candidates.shape) < self.mutation_rate
        mutated_population = np.where(mask, 1-crossovered_candidates, crossovered_candidates)

        return mutated_population

    def replace(self, mutated_candidates: np.ndarray) -> None:
        """
        Replace the old population with the new population.
        """
        self.population = mutated_candidates

    def get_best_candidate(self) -> np.ndarray:
        """
        Return the best candidate from the population.
        """
        fitness_values = self.evaluate_population()
        best_candidate_index = np.argmax(fitness_values)
        best_candidate = self.population[best_candidate_index]
        return best_candidate
