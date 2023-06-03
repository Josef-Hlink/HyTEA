#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from time import perf_counter
from pathlib import Path

from hytea.utils import DotDict
from hytea import Agent, Environment, Model

from yaml import safe_load
import torch


def cli() -> None:
    
    with open(Path(__file__).resolve().parents[0] / 'config.yaml', 'r') as f:
        blueprint = DotDict.from_dict(safe_load(f))

    print(blueprint)

    device = torch.device('cpu')
    env = Environment(env_name='LunarLander-v2', device=device)
    model = Model(
        input_size = env.observation_space.shape[0],
        output_size = env.action_space.n,
        hidden_size = blueprint.network.hidden_size.default,
        hidden_activation = blueprint.network.hidden_activation.default,
        num_layers = blueprint.network.num_layers.default,
        dropout_rate = blueprint.network.dropout_rate.default,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=blueprint.optimizer.lr.default)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=blueprint.optimizer.lr_step.default, gamma=blueprint.optimizer.lr_decay.default)
    agent = Agent(
        model = model,
        optimizer = optimizer,
        scheduler = scheduler,
        gamma = blueprint.agent.gamma.default,
        ent_reg_weight = blueprint.agent.ent_reg_weight.default,
        bl_sub = blueprint.agent.bl_sub.default,
        device = device
    )

    start = perf_counter()
    agent.train(num_episodes=1000, env=env)
    print(f'Training took {perf_counter() - start:.2f} seconds.')
    test_reward = agent.test(num_episodes=100)
    print(f'Test reward: {test_reward:.2f}')

    return


if __name__ == '__main__':
    cli()
