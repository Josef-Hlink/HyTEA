from time import perf_counter

from hytea import Environment, Model, Agent
from hytea.bitstring import BitStringDecoder
from hytea.wblog import WandbLogger, create_job_type_name
from hytea.utils import DotDict

import torch
import numpy as np


class FitnessFunction:

    def __init__(self,
        decoder: BitStringDecoder,
        env_name: str,
        num_train_episodes: int,
        num_test_episodes: int,
        num_runs: int,
        args: DotDict,
        debug: bool = False,
    ) -> None:
        """ Fitness function for the bitstrings.
        
        ### Args:
        `BitStringDecoder` decoder: The decoder to use.
        `str` env_name: The name of the environment to use.
        `int` num_train_episodes: The number of episodes to train for.
        `int` num_test_episodes: The number of episodes to test for.
        `int` num_runs: The number of individual runs to average over.
        """
        self.decoder = decoder
        self.env_name = env_name
        self.num_train_episodes = num_train_episodes
        self.num_test_episodes = num_test_episodes
        self.num_runs = num_runs
        self.D = debug
        self.args = args
        self.device = torch.device('cpu')
        return

    def evaluate(self, bitstring: np.ndarray, group_name: str, job_type_name: str) -> float:
        """ Run the agent for a number of episodes.

        ### Args:
        `np.ndarray` bitstring: The bitstring to evaluate.

        ### Returns:
        `float`: reward per episode, averaged over num_episodes and num_runs.
        """

        config = self.decoder.decode(bitstring)

        if self.D: print(f'Config: {config}')

        res = sum(self.evaluate_single(config, group_name, job_type_name) for _ in range(self.num_runs)) / self.num_runs

        print(f'{bitstring} -> {res}')
        print('-' * 80)

        return res

    def evaluate_single(self, config: DotDict, group_name: str, job_type_name: str) -> float:
        """ Helper (one run) """
        if self.args.use_wandb:
            config.update(**self.args, group_name=group_name, job_type=job_type_name)
            logger = WandbLogger(
                config = config,
            )

        env = Environment(env_name=self.env_name, device=self.device)

        model = Model(
            input_size = env.observation_space.shape[0],
            output_size = env.action_space.n,
            hidden_size = config.network.hidden_size,
            hidden_activation = config.network.hidden_activation,
            num_layers = config.network.num_layers,
            dropout_rate = config.network.dropout_rate,
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.optimizer.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.optimizer.lr_decay)

        agent = Agent(
            model = model,
            optimizer = optimizer,
            scheduler = scheduler,
            gamma = config.agent.gamma,
            n_steps = config.agent.n_steps,
            device = self.device
        )

        start = perf_counter()
        history = agent.train(num_episodes=self.num_train_episodes, env=env)
        
        if self.args.use_wandb:
            for i, h in enumerate(history):
                logger.log({'train_reward': h}, step=i)
        
        end = perf_counter()
        if self.D: print(f'Training took {end - start} seconds.')
        test_reward = agent.test(num_episodes=self.num_test_episodes)
        
        if self.args.use_wandb: 
            logger.update_summary({'test_reward': test_reward})
            logger.commit()
            logger.finish()
        
        return test_reward