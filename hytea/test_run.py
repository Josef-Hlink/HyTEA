from hytea.bitstring import BitStringDecoder
from hytea.wblog import WandbLogger, create_job_type_name, create_random_name
from hytea.utils import DotDict
from hytea.fitness import FitnessFunction

import torch
import numpy as np

class SingleEvaluation:
    """
    Single evaluation of a configuration.
    Config is a DotDict containing the configuration which is passed to _evaluate_single..
    We create the config here and pass it to the fitness function.
    Is used for testing purposes.
    """

    def __init__(self,
        env_name: str,
        num_train_episodes: int,
        num_test_episodes: int,
        num_runs: int,
        args: DotDict,
        debug: bool = False,
    ) -> None:
        """ 
        
        ### Args:
        `str` env_name: The name of the environment to use.
        `int` num_train_episodes: The number of episodes to train for.
        `int` num_test_episodes: The number of episodes to test for.
        `int` num_runs: The number of individual runs to average over.
        """
        self.env_name = env_name
        self.num_train_episodes = num_train_episodes
        self.num_test_episodes = num_test_episodes
        self.num_runs = num_runs
        self.D = debug
        self.args = args
        self.device = torch.device('cpu')
        return
    
    def config_mock(self) -> DotDict:
        """ Create a mock configuration. """
        config = DotDict()
        config.agent = DotDict()
        config.network = DotDict()
        config.optimizer = DotDict()
        config.agent.gamma = 0.85
        config.agent.n_steps = 100
        config.network.num_layers = 2
        config.network.dropout_rate = 0
        config.network.hidden_size = 64
        config.network.hidden_activation = 'relu'
        config.optimizer.lr = 0.02
        config.optimizer.lr_decay = 0.99
        config.lr_step = 1000
        
        config.epsilon = 0.1
        
        
        config.project_name = 'test'
        
        config.wandb_team = 'hytea'
        # config.group_name = 'test'
        # config.job_type = 'test'
        config.use_wandb = True
        config.num_generations = 10
        config.population_size = 10
        
        config.num_episodes = 1000
        config.num_test_episodes = 100
        config.num_runs = 200
        
        config.debug = True
        config.device = self.device
        config.env_name = self.env_name
        config.num_train_episodes = self.num_train_episodes
        config.num_test_episodes = self.num_test_episodes
        config.num_runs = self.num_runs
        
        return config
    
    def evaluate(self, config: DotDict, group_name: str, job_type_name: str) -> float:
        """ Run the agent for a number of episodes.

        ### Args:
        `np.ndarray` bitstring: The bitstring to evaluate.

        ### Returns:
        `float`: reward per episode, averaged over num_episodes and num_runs.
        """
        decoder = BitStringDecoder(config)
        
        fitness_function = FitnessFunction(decoder, self.env_name, self.num_train_episodes, self.num_test_episodes, self.num_runs, self.args, self.D)
        
        res = sum(fitness_function.evaluate_single(config, group_name, job_type_name) for _ in range(self.num_runs)) / self.num_runs

        print(f'{config} -> {res}')
        print('-' * 80)

        return res
    
    
if __name__ == '__main__':
    args = DotDict()
    args.use_wandb = True
    
    job_type_name = create_random_name()
    
    se = SingleEvaluation('LunarLander-v2', 1000, 100, 1, args, True)
    se.evaluate(se.config_mock(), 'test', job_type_name)