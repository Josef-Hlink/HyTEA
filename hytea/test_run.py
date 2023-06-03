from hytea.bitstring import BitStringDecoder
from hytea.wblog import WandbLogger, create_job_type_name, create_random_name
from hytea.utils import DotDict
from hytea.fitness import FitnessFunction

import torch
import numpy as np
import argparse
import sys
    
class SingleEvaluation:
    """
    Single evaluation of a configuration.
    Config is a DotDict containing the configuration which is passed to _evaluate_single..
    We create the config here and pass it to the fitness function.
    Is used for testing purposes.
    """

    def __init__(self,
        args: DotDict
    ) -> None:
        """ 
        
        ### Args:
        `str` env_name: The name of the environment to use.
        `int` num_train_episodes: The number of episodes to train for.
        `int` num_test_episodes: The number of episodes to test for.
        `int` num_runs: The number of individual runs to average over.
        """
        self.env_name = args.env_name
        self.num_train_episodes = args.num_train_episodes
        self.num_test_episodes = args.num_test_episodes
        self.num_runs = args.num_runs
        self.D = args.debug
        self.args = args
        self.device = torch.device('cpu')
        return
    
    def evaluate(self, group_name: str, job_type_name: str) -> float:
        """ Run the agent for a number of episodes.

        ### Args:
        `np.ndarray` bitstring: The bitstring to evaluate.

        ### Returns:
        `float`: reward per episode, averaged over num_episodes and num_runs.
        """
        del self.args.group_name
        decoder = BitStringDecoder(self.args)
        
        fitness_function = FitnessFunction(self.args, decoder)
        
        res = sum(fitness_function.evaluate_single(self.args, group_name, job_type_name) for _ in range(self.num_runs)) / self.num_runs

        print(f'{self.args.optimizer}')
        print(f'{self.args.network}')
        print(f'{self.args.agent}')
        print(f'->{res}')
        print('-' * 80)

        return res
    
def main():
    """ Command line interface for SingleEvaluation. """

    parser = argparse.ArgumentParser(description='Evaluate a bitstring.')
    parser.add_argument('--env_name', type=str, default='LunarLander-v2', help='The environment to use.')
    parser.add_argument('--num_train_episodes', type=int, default=1000, help='The number of episodes to train for.')
    parser.add_argument('--num_test_episodes', type=int, default=100, help='The number of episodes to test for.')
    parser.add_argument('--num_runs', type=int, default=1, help='The number of individual runs to average over.')
    parser.add_argument('--debug', action='store_true', help='Whether to run in debug mode.')
    parser.add_argument('-W', dest='use_wandb', action='store_true', help='Use wandb for logging.')
    parser.add_argument('--gamma', type=float, default=0.99, help='The discount factor.')
    parser.add_argument('--n_steps', type=int, default=100, help='The number of steps to look ahead.')
    parser.add_argument('--num_layers', type=int, default=1, help='The number of hidden layers.')
    parser.add_argument('--dropout_rate', type=float, default=0, help='The dropout rate.')
    parser.add_argument('--hidden_size', type=int, default=64, help='The size of the hidden layers.')
    parser.add_argument('--hidden_activation', type=str, default='relu', help='The activation function to use.')
    parser.add_argument('--lr', type=float, default=0.02, help='The learning rate.')
    parser.add_argument('--lr_decay', type=float, default=0.99, help='The learning rate decay.')
    parser.add_argument('--lr_step', type=int, default=1000, help='The learning rate step.')
    parser.add_argument('--project_name', type=str, default='test', help='The name of the project.')
    parser.add_argument('--wandb_team', type=str, default='hytea', help='The name of the wandb team.')
    parser.add_argument('--num_generations', type=int, default=10, help='The number of generations.')
    parser.add_argument('--population_size', type=int, default=10, help='The size of the population.')
    parser.add_argument('--num_episodes', type=int, default=1000, help='The number of episodes to train for.')
    parser.add_argument('--device', type=str, default='cpu', help='The device to use.')
    parser.add_argument('--group_name', type=str, default='test', help='The name of the group.')
    args = parser.parse_args()
    
    # convert args to DotDict
    args = DotDict(vars(args))
    
    # now for agent, network and optimizer we need to create a DotDict and add it to args
    args.agent = DotDict()
    args.network = DotDict()
    args.optimizer = DotDict()
    args.agent.gamma = args.gamma
    args.agent.n_steps = args.n_steps
    args.network.num_layers = args.num_layers
    args.network.dropout_rate = args.dropout_rate
    args.network.hidden_size = args.hidden_size
    args.network.hidden_activation = args.hidden_activation
    args.optimizer.lr = args.lr
    args.optimizer.lr_decay = args.lr_decay
    args.optimizer.lr_step = args.lr_step
    
    se = SingleEvaluation(args)
    
    se.evaluate(args.group_name, create_random_name())
    
    return


def cli():
    main()
    
    
if __name__ == '__main__':
    main()
