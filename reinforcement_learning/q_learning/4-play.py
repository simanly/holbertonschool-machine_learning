#!/usr/bin/env python3
"""
Module 4-play
Defines the function play that has the trained agent play an episode.
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode using greedy strategy.

    Parameters:
    - env: FrozenLakeEnv instance
    - Q: numpy.ndarray containing the Q-table
    - max_steps: maximum number of steps in the episode

    Returns:
    - total_reward: the total rewards for the episode
    - rendered_outputs: list of rendered outputs representing board states
    """
    state, _ = env.reset()
    rendered_outputs = [env.render()]

    total_reward = 0

    for _ in range(max_steps):
        action = np.argmax(Q[state])
        state, reward, terminated, truncated, _ = env.step(action)

        rendered_outputs.append(env.render())
        total_reward += reward

        if terminated or truncated:
            break

    return total_reward, rendered_outputs
