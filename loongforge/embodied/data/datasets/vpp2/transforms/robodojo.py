# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Preserve every RoboDojo sample field, including history and pad masks."""

from dataclasses import dataclass

from torch.utils.data import default_collate

from loongforge.embodied.data.datasets.transforms.collator import (
    BasePreprocessor,
    PreparedBatch,
    register_preprocessor,
)
from loongforge.embodied.data.datasets.transforms.registry import register_transform_builder


@dataclass
class RoboDojoBatch(PreparedBatch):
    sample: dict

    @property
    def action(self):
        return self.sample["action"]

    def to_sample(self):
        return self.sample


@register_preprocessor("vpp2_robodojo")
class RoboDojoPreprocessor(BasePreprocessor):
    @classmethod
    def from_config(cls, *args, **kwargs):
        return cls()

    def __call__(self, examples):
        return RoboDojoBatch(default_collate(examples))


@register_transform_builder("vpp2_robodojo")
def build_robodojo_transforms(context):
    # The transplanted dataset already applies the exact EE16 z-score pipeline.
    return []
