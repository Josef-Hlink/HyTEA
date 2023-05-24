from time import perf_counter

from hytea import Environment, Model, Agent
from hytea.bitstring import BitStringDecoder

import torch
import numpy as np


class FitnessFunction:

    def __init__(self, decoder: BitStringDecoder) -> None:
        self.decoder = decoder

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

        def _evaluate(num_episodes: int) -> float:
            """ Helper (one run) """
            env = Environment('LunarLander-v2', device=device)

            model = Model(
                input_size = env.observation_space.shape[0],
                output_size = env.action_space.n,
                hidden_size = config.network.hidden_size,
                hidden_activation = config.network.hidden_activation,
                num_layers = config.network.num_layers,
                dropout_rate = config.network.dropout_rate,
            ).to(device)

            optimizer: torch.optim.Adam(model.parameters(), lr=config.optimizer.lr)
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
            agent.train(num_episodes=num_train_episodes, env=env)
            end = perf_counter()
            print(f'Training took {end - start} seconds.')
            return agent.test(num_episodes=num_test_episodes)

        # HARDCODED (for now)
        num_runs = 3
        num_train_episodes = 1000
        num_test_episodes = 100
        
        res = sum(_evaluate(num_train_episodes, num_test_episodes) for _ in range(num_runs)) / num_runs

        print(f'{bitstring} -> {res}')
        print('-' * 80)

        return res
