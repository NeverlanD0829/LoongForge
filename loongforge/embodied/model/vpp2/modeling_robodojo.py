# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Joint Video + Action2B RoboDojo training on the native embodied trainer."""

from dataclasses import dataclass, field

import torch

from loongforge.embodied.model.registry import register_model

from .modeling_configuration_vpp2 import VPP2ModelConfig
from .modeling_vpp2 import VPP2Model
from .modules.vpp2 import VPP2


@dataclass(frozen=True)
class RoboDojoModelConfig(VPP2ModelConfig):
    model_type: str = "vpp2_robodojo"
    action_dim: int = 16
    proprio_dim: int = 16
    action_visible_video_frames: int = 5
    action_only_first_frame_fast_path: bool = False
    action_dit_config: dict = field(
        default_factory=lambda: {
            **VPP2ModelConfig().action_dit_config,
            "action_dim": 16,
            "hidden_dim": 1024,
            "ffn_dim": 4096,
        }
    )
    action_scheduler: dict = field(
        default_factory=lambda: {
            "train_shift": 3.0,
            "infer_shift": 3.0,
            "num_train_timesteps": 1000,
        }
    )

    def __post_init__(self):
        if self.model_type != "vpp2_robodojo":
            raise ValueError("Expected the RoboDojo joint model")
        if (
            self.action_dim,
            self.proprio_dim,
            self.action_dit_config["hidden_dim"],
            self.action_dit_config["ffn_dim"],
        ) != (16, 16, 1024, 4096):
            raise ValueError("RoboDojo requires EE16, proprio16 and Action2B")


@register_model("vpp2_robodojo")
class RoboDojoModel(VPP2):
    config_class = RoboDojoModelConfig
    _no_split_modules = VPP2Model._no_split_modules
    default_fp8_targets = staticmethod(VPP2Model.default_fp8_targets)
    default_lora_targets = staticmethod(VPP2Model.default_lora_targets)
    load_pretrained = VPP2Model.load_pretrained
    on_after_data_iterators_initialized = VPP2Model.on_after_data_iterators_initialized

    @classmethod
    def from_pretrained(cls, cfg):
        return VPP2Model.from_pretrained.__func__(cls, cfg)

    def set_frozen_modules_to_eval_mode(self):
        for module in (self.vae, self.clip_encoder, self.text_encoder):
            if module is not None:
                module.eval()

    def freeze_modules(self):
        self.requires_grad_(False)
        self.mot.train().requires_grad_(True)
        if self.proprio_encoder is not None:
            self.proprio_encoder.train().requires_grad_(True)
        self.set_frozen_modules_to_eval_mode()

    def train(self, mode=True):
        # BASE keeps the outer model in eval and enables its two experts only.
        torch.nn.Module.train(self, False)
        self.mot.train(mode)
        if self.proprio_encoder is not None:
            self.proprio_encoder.train(mode)
        self.set_frozen_modules_to_eval_mode()
        return self

    def forward(self, batch, iteration=None, **kwargs):
        sample = batch.to_sample() if hasattr(batch, "to_sample") else batch
        loss, metrics = self.training_loss(sample)
        observed = {"loss": loss.detach(), "video_loss": metrics["loss_video"], "action_loss": metrics["loss_action"]}
        return loss, observed
