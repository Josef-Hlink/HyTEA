#!/usr/bin/env python

"""
Evolutionary Algorithm
This module contains the evolutionary algorithm class and its components.
"""

import numpy as np
from hytea.bitstring import BitStringDecoder
from hytea.utils import DotDict
from hytea.main import run

def main(config: DotDict, caller: str = 'cli') -> None | list:
    bs = BitStringDecoder(50, 'hytea/bitstring.yaml')
    ea = EvolutionaryAlgorithm(50, bs, config)
    ea.run()

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

    def __init__(self, population_size: int, bitstring_decoder: BitStringDecoder, default_config: DotDict) -> None:
        self.population_size = population_size
        self.candidate_size = bitstring_decoder.candidate_size
        self.mu_ = population_size // 5
        self.lambda_ = population_size - self.mu_
        self.pool_size = population_size // 4
        self.mutation_rate = 0.1
        self.generations = 50
        self.bitstring_decoder = bitstring_decoder
        self.default_config = default_config
    
    def run(self) -> None:
        """
        Run the evolutionary algorithm.
        """
        self.initialize_population()

        for i in range(self.generations):
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
        """
        Evaluate the population by training and testing the bitstrings.
        """        
        decoded_population = np.array([self.bitstring_decoder.decode(bitstring, self.default_config.to_dict()) for bitstring in self.population])
 
        histories = np.array([run(config=DotDict.from_dict(decoded_config)) for decoded_config in decoded_population])
        
        # fitness values are the average reward per episode
        fitness_values = np.mean(histories, axis=1)
        
        return fitness_values

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

if __name__=="__main__":
    main()
