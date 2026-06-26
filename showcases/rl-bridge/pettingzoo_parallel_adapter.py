"""Optional PettingZoo ParallelEnv adapter for GrADyS-backed MARL."""

from __future__ import annotations

from typing import Optional

from gradys_uav_service_env import GradysUAVServiceCoreEnv, UAVServiceEnvConfig

try:
    import numpy as np
    from gymnasium import spaces
    from pettingzoo import ParallelEnv
except ImportError:  # pragma: no cover - optional dependency path
    np = None
    spaces = None
    ParallelEnv = object


class GradysUAVServiceParallelEnv(ParallelEnv):
    """PettingZoo parallel API wrapper.

    This wrapper is the natural bridge for MAPPO/IPPO-style libraries that
    consume parallel multi-agent environments.
    """

    metadata = {"name": "gradys_uav_service_v0", "render_modes": ["ansi"]}

    def __init__(self, config: Optional[UAVServiceEnvConfig] = None) -> None:
        if spaces is None:
            raise ImportError("Install pettingzoo and gymnasium to use this adapter.")
        self.core = GradysUAVServiceCoreEnv(config)
        self.possible_agents = list(self.core.agents)
        self.agents = list(self.possible_agents)
        self.observation_spaces = {
            agent: spaces.Box(
                low=-float("inf"),
                high=float("inf"),
                shape=(self.core.observation_size,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: spaces.Discrete(self.core.action_size)
            for agent in self.possible_agents
        }

    def reset(self, seed=None, options=None):
        self.agents = list(self.possible_agents)
        observations = self.core.reset(seed=seed)
        return self._as_arrays(observations), {agent: {} for agent in self.agents}

    def step(self, actions):
        result = self.core.step({agent: int(actions.get(agent, 0)) for agent in self.agents})
        observations = self._as_arrays(result.observations)
        terminations = {agent: result.terminated for agent in self.agents}
        truncations = {agent: result.truncated for agent in self.agents}
        rewards = {agent: float(result.rewards[agent]) for agent in self.agents}
        infos = result.infos
        if result.terminated or result.truncated:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def render(self):
        return self.core.render_text()

    def close(self):
        self.core.close()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _as_arrays(self, observations):
        return {
            agent: np.asarray(observations[agent], dtype=np.float32)
            for agent in observations
        }


def parallel_env(config: Optional[UAVServiceEnvConfig] = None):
    return GradysUAVServiceParallelEnv(config)
