from time import perf_counter

from hytea import Environment, Model, Agent
from hytea.bitstring import BitStringDecoder
from hytea.wblog import WandbLogger
from hytea.utils import DotDict

import torch
import numpy as np


class FitnessFunction:

    def __init__(self,
        args: DotDict,
        decoder: BitStringDecoder,
    ) -> None:
        """ Fitness function for the bitstrings.
        
        ### Args:
        `DotDict` args: The arguments to use for wandb logging.
        `BitStringDecoder` decoder: The decoder to use.
        """
        self.decoder = decoder
        self.env_name: str = args.env_name
        self.num_train_episodes: int = args.num_train_episodes
        self.num_test_episodes: int = args.num_test_episodes
        self.num_runs: int = args.num_runs
        self.D: bool = args.debug
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

        # gamma nstep num_layers dropout hidden_size hidden_activation lr lr_decay lr_step
        print(' gm  ns  nl  do  hs  a lr  ld  ls')
        print('    |   |   |   |   | |   |   |')
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
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.optimizer.lr_step, gamma=config.optimizer.lr_decay)

        agent = Agent(
            model = model,
            optimizer = optimizer,
            scheduler = scheduler,
            gamma = config.agent.gamma,
            ent_reg_weight = config.agent.ent_reg_weight,
            bl_sub = config.agent.bl_sub,
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
