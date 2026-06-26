"""RLlib old-stack Torch model for discrete action masking."""

from __future__ import annotations

from gymnasium.spaces import Dict as DictSpace
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN

torch, nn = try_import_torch()

ACTION_MASK_MODEL = "gradys_action_mask_model"


class GradysActionMaskTorchModel(TorchModelV2, nn.Module):
    """Apply an action mask to logits while learning from raw observations."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ) -> None:
        original_space = getattr(obs_space, "original_space", obs_space)
        if not (
            isinstance(original_space, DictSpace)
            and "action_mask" in original_space.spaces
            and "observations" in original_space.spaces
        ):
            raise ValueError(
                "GradysActionMaskTorchModel expects a Dict observation space "
                "with 'observations' and 'action_mask' entries."
            )

        TorchModelV2.__init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            **kwargs,
        )
        nn.Module.__init__(self)

        self.internal_model = TorchFC(
            original_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

    def forward(self, input_dict, state, seq_lens):
        action_mask = input_dict["obs"]["action_mask"]
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observations"]})
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        return logits + inf_mask, state

    def value_function(self):
        return self.internal_model.value_function()


def register_action_mask_model() -> None:
    """Register the custom model name used by training and checkpoint restore."""
    ModelCatalog.register_custom_model(ACTION_MASK_MODEL, GradysActionMaskTorchModel)
