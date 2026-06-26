"""Optional RLlib MultiAgentEnv adapter for the GrADyS RL bridge."""

from __future__ import annotations

from dataclasses import fields
from typing import Optional

from gradys_uav_service_env import GradysUAVServiceCoreEnv, UAVServiceEnvConfig

try:
    import numpy as np
    from gymnasium import spaces
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except ImportError:  # pragma: no cover - optional dependency path
    np = None
    spaces = None
    MultiAgentEnv = object


class GradysUAVServiceRLlibEnv(MultiAgentEnv):
    """RLlib multi-agent wrapper with one policy action per UAV."""

    def __init__(self, config: Optional[dict] = None) -> None:
        if spaces is None:
            raise ImportError("Install ray[rllib] and gymnasium to use this adapter.")
        config = dict(config or {})
        self._team_reward = bool(config.pop("rllib_team_reward", False))
        self._action_mask = bool(config.pop("rllib_action_mask", False))
        self._mask_hover_when_candidates = bool(
            config.pop("rllib_mask_hover_when_candidates", False)
        )
        self._auto_increment_seed = bool(config.pop("rllib_auto_increment_seed", True))
        self._base_seed = config.get("seed")
        self._episode_index = 0
        allowed_config_keys = {field.name for field in fields(UAVServiceEnvConfig)}
        env_config = UAVServiceEnvConfig(
            **{
                key: value
                for key, value in config.items()
                if key in allowed_config_keys
            }
        )
        self.core = GradysUAVServiceCoreEnv(env_config)
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
        result = self.core.step({agent: int(action_dict.get(agent, 0)) for agent in self.possible_agents})
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
                "action_mask": self._action_mask_for(agent),
            }
            for agent in observations
        }

    def _action_mask_for(self, agent):
        mask = np.zeros(self.core.action_size, dtype=np.float32)
        if self.core._busy_until[agent] > self.core.time or self.core._targets[agent] is not None:
            mask[0] = 1.0
            return mask

        candidates = self.core.candidate_tasks(agent)
        if not candidates:
            mask[0] = 1.0
            return mask

        if not self._mask_hover_when_candidates:
            mask[0] = 1.0
        mask[1 : len(candidates) + 1] = 1.0
        return mask
