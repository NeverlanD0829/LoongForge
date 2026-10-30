# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Register the native-frame EE16 pipeline and whole-batch sampler."""

from dataclasses import dataclass, field
from typing import Any

from hydra.utils import instantiate
from omegaconf import OmegaConf

from loongforge.embodied.data.datasets.sampler_builder import register_sampler_builder

from .samplers import build_vpp2_sampler


@dataclass(frozen=True)
class RoboDojoDataConfig:
    pipeline: dict[str, Any] = field(default_factory=dict)
    model_cfg: Any = field(default=None, repr=False, compare=False)


def build_robodojo_dataset(model_cfg, data_cfg, training_args):
    return instantiate(OmegaConf.create(data_cfg.pipeline))


register_sampler_builder("vpp2_robodojo")(build_vpp2_sampler)
