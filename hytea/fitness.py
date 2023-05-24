from time import perf_counter

from hytea import Environment, Model, Agent
from hytea.bitstring import BitStringDecoder

import torch
import numpy as np


class FitnessFunction:

    def __init__(self,
        decoder: BitStringDecoder,
        env_name: str,
        num_train_episodes: int,
        num_test_episodes: int,
        num_runs: int,
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
        return

    def evaluate(self, bitstring: np.ndarray) -> float:
        """ Run the agent for a number of episodes.

        ### Args:
        `np.ndarray` bitstring: The bitstring to evaluate.

        ### Returns:
        `float`: reward per episode, averaged over num_episodes and num_runs.
        """

        bitstring = bitstring.astype(np.int8)

        config = self.decoder.decode(bitstring)

        print(f'Config: {config}')
        
        device = torch.device('cpu')

        def _evaluate() -> float:
            """ Helper (one run) """
            env = Environment(env_name=self.env_name, device=device)

            model = Model(
                input_size = env.observation_space.shape[0],
                output_size = env.action_space.n,
                hidden_size = config.network.hidden_size,
                hidden_activation = config.network.hidden_activation,
                num_layers = config.network.num_layers,
                dropout_rate = config.network.dropout_rate,
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=config.optimizer.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.optimizer.lr_decay)

            agent = Agent(
                model = model,
                optimizer = optimizer,
                scheduler = scheduler,
                gamma = config.agent.gamma,
                n_steps = config.agent.n_steps,
                device = device
            )

            start = perf_counter()
            agent.train(num_episodes=self.num_train_episodes, env=env)
            end = perf_counter()
            print(f'Training took {end - start} seconds.')
            return agent.test(num_episodes=self.num_test_episodes)
        
        res = sum(_evaluate() for _ in range(self.num_runs)) / self.num_runs

        print(f'{bitstring} -> {res}')
        print('-' * 80)

        return res
