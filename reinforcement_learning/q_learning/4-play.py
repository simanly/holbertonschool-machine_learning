#!/usr/bin/env python3
"""
Module 4-play
Defines the function play that has the trained agent play an episode.
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode using a greedy policy.

    Parameters:
    - env: FrozenLakeEnv instance
    - Q: numpy.ndarray containing the Q-table
    - max_steps: maximum number of steps in the episode

    Returns:
    - total_reward: total rewards for the episode
    - rendered_outputs: list of rendered outputs representing the board state
    """
    reset_res = env.reset()
    state = reset_res[0] if isinstance(reset_res, tuple) else reset_res

    rendered_outputs = [env.render()]
    total_reward = 0

    for _ in range(max_steps):
        action = np.argmax(Q[state])
        step_res = env.step(action)

        if len(step_res) == 5:
            state, reward, terminated, truncated, _ = step_res
            done = terminated or truncated
        else:
            state, reward, done, _ = step_res

        rendered_outputs.append(env.render())
        total_reward += reward

        if done:
            break

    return total_reward, rendered_outputs
