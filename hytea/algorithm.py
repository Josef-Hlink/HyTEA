from multiprocessing import Pool

from hytea.fitness import FitnessFunction
from hytea.wblog import create_group_name, create_job_type_name
from hytea.utils import DotDict
import numpy as np


class EvolutionaryAlgorithm:
    """
    Evolutionary Algorithm to recommend different neural networks.
    The bitstrings will represent the neural network architecture and its hyperparameters.
    """

    def __init__(self,
            num_generations: int,
            population_size: int,
            fitness_function: FitnessFunction,
            args: DotDict,
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
        self.args = args

        return
    
    def run(self) -> None:
        """ Main loop of the evolutionary algorithm. """
        self.initialize_population()

        for i in range(self.num_generations):
            fitness_values = self.evaluate_population(i+1)
            print(f'Generation {i}: {np.mean(fitness_values)}')
            print([round(f, 2) for f in fitness_values])
            selected_candidates = self.select(fitness_values)
            crossovered_candidates = self.crossover(selected_candidates)
            mutated_candidates = self.mutate(crossovered_candidates)
            self.replace(mutated_candidates)
        
        best_candidate = self.population[np.argmax(fitness_values)]
        print(f'BEST: {best_candidate} -> {np.max(fitness_values)}')

    def initialize_population(self) -> None:
        """ Initialize the population with random bitstrings. """
        self.population = np.random.randint(2, size=(self.population_size, self.candidate_size), dtype=np.uint8)
    
    def evaluate_population(self, gen: int) -> np.ndarray:
        """ Evaluate the population by training and testing the bitstrings. """
        group_name = create_group_name(self.args, gen)
        with Pool() as pool:
            return np.array(pool.starmap(self.evaluate, [(candidate, group_name, create_job_type_name(candidate)) for candidate in self.population]))
        
    def select(self, fitness_values: np.ndarray) -> np.ndarray:
        """ Select the best candidates from the population using tournament selection. """
        selected_candidates = np.empty((self.mu_, self.candidate_size), dtype=np.uint8)

        for i in range(self.mu_):
            tournament_indices = np.random.choice(len(self.population), self.pool_size, replace=False)
            tournament = self.population[tournament_indices]
            best_individual_index = np.argmax(fitness_values[tournament_indices])
            best_individual = tournament[best_individual_index]
            selected_candidates[i] = best_individual

        return selected_candidates


    def crossover(self, selected_candidates: np.ndarray) -> np.ndarray:
        """ Perform crossover on the selected candidates. """
        crossovered_candidates = np.empty((self.lambda_, selected_candidates.shape[1]), dtype=np.uint8)

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
        """ Mutate the crossovered candidates. """
        mask = np.random.rand(*crossovered_candidates.shape) < self.mutation_rate
        mutated_population = np.where(mask, 1-crossovered_candidates, crossovered_candidates)

        return mutated_population

    def replace(self, mutated_candidates: np.ndarray) -> None:
        """ Replace the old population with the new population. """
        self.population = mutated_candidates
