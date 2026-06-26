"""Optional centralized Gymnasium adapter for the GrADyS RL bridge."""

from __future__ import annotations

from typing import Optional

from gradys_uav_service_env import (
    GradysUAVServiceCoreEnv,
    UAVServiceEnvConfig,
    flatten_agent_dict,
)

try:
    import gymnasium as gym
    import numpy as np
    from gymnasium import spaces
except ImportError:  # pragma: no cover - optional dependency path
    gym = None
    np = None
    spaces = None


class GradysUAVServiceGymEnv(gym.Env if gym is not None else object):
    """Centralized single-agent view over all UAVs.

    The action is a MultiDiscrete vector with one task-choice action per UAV.
    This is convenient for PPO baselines and quick smoke tests. For MAPPO-style
    learning, prefer the PettingZoo or RLlib adapters.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, config: Optional[UAVServiceEnvConfig] = None) -> None:
        if gym is None:
            raise ImportError("Install gymnasium to use GradysUAVServiceGymEnv.")
        self.core = GradysUAVServiceCoreEnv(config)
        self.action_space = spaces.MultiDiscrete([self.core.action_size] * len(self.core.agents))
        obs_size = self.core.observation_size * len(self.core.agents)
        self.observation_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=(obs_size,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        observations = self.core.reset(seed=seed)
        return self._flatten(observations), {}

    def step(self, action):
        action_dict = {
            agent: int(action[index])
            for index, agent in enumerate(self.core.agents)
        }
        result = self.core.step(action_dict)
        reward = float(sum(result.rewards.values()))
        info = {"agent_infos": result.infos}
        return self._flatten(result.observations), reward, result.terminated, result.truncated, info

    def render(self):
        return self.core.render_text()

    def close(self):
        self.core.close()

    def _flatten(self, observations):
        return np.asarray(flatten_agent_dict(observations, self.core.agents), dtype=np.float32)
