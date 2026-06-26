"""Optional RLlib MultiAgentEnv adapter for the uavsim4security bridge."""

from __future__ import annotations

from dataclasses import fields
from typing import Optional

from uavsim4security_core_env import UAVSim4SecurityCoreEnv, UAVSim4SecurityEnvConfig

try:
    import numpy as np
    from gymnasium import spaces
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except ImportError:  # pragma: no cover - optional dependency path
    np = None
    spaces = None
    MultiAgentEnv = object


class UAVSim4SecurityRLlibEnv(MultiAgentEnv):
    """RLlib multi-agent wrapper with one shared routing policy for UAVs."""

    def __init__(self, config: Optional[dict] = None) -> None:
        if spaces is None:
            raise ImportError("Install ray[rllib] and gymnasium to use this adapter.")
        config = dict(config or {})
        self._team_reward = bool(config.pop("rllib_team_reward", True))
        self._action_mask = bool(config.pop("rllib_action_mask", True))
        self._auto_increment_seed = bool(config.pop("rllib_auto_increment_seed", True))
        self._base_seed = config.get("seed")
        self._episode_index = 0
        allowed_config_keys = {field.name for field in fields(UAVSim4SecurityEnvConfig)}
        env_config = UAVSim4SecurityEnvConfig(
            **{
                key: value
                for key, value in config.items()
                if key in allowed_config_keys
            }
        )
        self.core = UAVSim4SecurityCoreEnv(env_config)
        self._agent_ids = set(self.core.agents)
        self.agents = list(self.core.agents)
        self.possible_agents = list(self.core.agents)
        super().__init__()
        self._raw_observation_space = spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(self.core.observation_size,),
            dtype=np.float32,
        )
        if self._action_mask:
            self.observation_space = spaces.Dict(
                {
                    "action_mask": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(self.core.action_size,),
                        dtype=np.float32,
                    ),
                    "observations": self._raw_observation_space,
                }
            )
        else:
            self.observation_space = self._raw_observation_space
        self.action_space = spaces.Discrete(self.core.action_size)
        self.observation_spaces = {
            agent: self.observation_space
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: self.action_space
            for agent in self.possible_agents
        }

    def reset(self, *, seed=None, options=None):
        self.agents = list(self.possible_agents)
        if seed is None and self._auto_increment_seed and self._base_seed is not None:
            seed = int(self._base_seed) + self._episode_index
            self._episode_index += 1
        observations = self.core.reset(seed=seed)
        return self._as_observations(observations), {agent: {} for agent in self.possible_agents}

    def step(self, action_dict):
        result = self.core.step(
            {
                agent: int(action_dict.get(agent, 0))
                for agent in self.possible_agents
            }
        )
        observations = self._as_observations(result.observations)
        if self._team_reward:
            team_reward = sum(result.rewards.values()) / max(1, len(self.possible_agents))
            rewards = {agent: float(team_reward) for agent in self.possible_agents}
        else:
            rewards = {agent: float(result.rewards[agent]) for agent in self.possible_agents}
        terminateds = {agent: result.terminated for agent in self.possible_agents}
        truncateds = {agent: result.truncated for agent in self.possible_agents}
        terminateds["__all__"] = result.terminated
        truncateds["__all__"] = result.truncated
        if result.terminated or result.truncated:
            self.agents = []
        return observations, rewards, terminateds, truncateds, result.infos

    def close(self):
        self.core.close()

    def _as_observations(self, observations):
        if not self._action_mask:
            return {
                agent: np.asarray(observations[agent], dtype=np.float32)
                for agent in observations
            }
        return {
            agent: {
                "observations": np.asarray(observations[agent], dtype=np.float32),
                "action_mask": np.asarray(self.core.action_mask(agent), dtype=np.float32),
            }
            for agent in observations
        }
