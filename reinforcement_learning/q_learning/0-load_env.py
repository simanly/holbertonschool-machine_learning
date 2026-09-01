#!/usr/bin/env python3
"""
Module 0-load_env
Defines the function load_frozen_lake to load
the Gymnasium FrozenLakeEnv.
"""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the pre-made FrozenLakeEnv environment from gymnasium.

    Parameters:
    - desc: list of lists with custom map description, or None
    - map_name: string containing pre-made map name, or None
    - is_slippery: boolean determining if ice is slippery

    Returns:
    - env: loaded gymnasium environment
    """
    if desc is None and map_name is None:
        map_name = '8x8'

    env = gym.make(
        'FrozenLake-v1',
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery
    )
    return env
