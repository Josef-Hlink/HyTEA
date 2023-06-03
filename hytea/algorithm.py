from multiprocessing import Pool

from hytea.fitness import FitnessFunction
from hytea.utils.wblog import create_group_name, create_job_type_name
from hytea.utils import DotDict
import numpy as np


class EvolutionaryAlgorithm:
    """
    Evolutionary Algorithm to recommend different neural networks.
    The bitstrings will represent the neural network architecture and its hyperparameters.
    """

    def __init__(self,
            args: DotDict,
            fitness_function: FitnessFunction,
    ) -> None:
        """ Initialize the evolutionary algorithm.

        ### Args:
        `DotDict` args: The arguments to use for wandb logging.
        `FitnessFunction` fitness_function: The fitness function to evaluate the candidates with.
            This will be used to evaluate the population, and to get the size of a candidate bitstring.
        """

        self.num_generations: int = args.num_generations
        self.population_size: int = args.population_size
        
        self.mu_: int = args.population_size // 2
        self.lambda_: int = args.population_size - self.mu_
        self.mutation_rate = 0.1

        self.evaluate = fitness_function.evaluate
        self.candidate_size = fitness_function.decoder.get_candidate_size()
        self.args = args

        return
    
    def run(self) -> None:
        """ Main loop of the evolutionary algorithm. """
        self.initialize_population()
        best_candidate = None
        best_fitness = -np.inf

        for i in range(self.num_generations):
            fitness_values = self.evaluate_population(i+1)
            print(f'Generation {i}: {np.mean(fitness_values)}')
            print([round(f, 2) for f in fitness_values])
            if np.max(fitness_values) > best_fitness:
                best_fitness = np.max(fitness_values)
                best_candidate = self.population[np.argmax(fitness_values)]
            selected_candidates = self.select(fitness_values)
            crossovered_candidates = self.crossover(selected_candidates)
            self.population = np.concatenate((selected_candidates, crossovered_candidates))
            self.mutate()
        
        print(f'BEST: {best_candidate} -> {best_fitness}')
        return

    def initialize_population(self) -> None:
        """ Initialize the population with random bitstrings. """
        self.population = np.random.randint(2, size=(self.population_size, self.candidate_size), dtype=np.uint8)
    
    def evaluate_population(self, gen: int) -> np.ndarray:
        """ Evaluate the population by training and testing the bitstrings. """
        group_name = create_group_name(self.args, gen)
        with Pool() as pool:
            return np.array(pool.starmap(self.evaluate, [(candidate, group_name, create_job_type_name(candidate)) for candidate in self.population]))
        
    def select(self, fitness_values: np.ndarray) -> np.ndarray:
        """ Select the mu_ best candidates from the population. """
        selected_candidates = np.empty((self.mu_, self.candidate_size), dtype=np.uint8)

        for i in range(self.mu_):
            best_candidate_index = np.argmax(fitness_values)
            selected_candidates[i] = self.population[best_candidate_index]
            fitness_values[best_candidate_index] = -np.inf

        return selected_candidates

    def crossover(self, selected_candidates: np.ndarray) -> np.ndarray:
        """ Perform crossover on the selected candidates. """
        crossovered_candidates = np.empty((self.lambda_, selected_candidates.shape[1]), dtype=np.uint8)

        for i in range(0, self.lambda_):
            parent1 = selected_candidates[np.random.randint(self.mu_)]
            parent2 = selected_candidates[np.random.randint(self.mu_)]
            crossover_point = np.random.randint(self.candidate_size)
            crossovered_candidates[i] = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))

        return crossovered_candidates

    def mutate(self) -> None:
        """ Mutate the crossovered candidates. """
        mask = np.random.rand(*self.population.shape) < self.mutation_rate
        self.population = np.where(mask, 1-self.population, self.population)
        return
