# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""RoboDojo framework contracts; full-model parity is tested by native runs."""

from types import SimpleNamespace

import pytest
import torch

from loongforge.embodied.data.datasets.vpp2.transforms.robodojo import RoboDojoPreprocessor
from loongforge.embodied.model import build_model
from loongforge.embodied.model.vpp2.modeling_robodojo import RoboDojoModelConfig
from loongforge.embodied.train.trainers.custom.vpp2.robodojo_trainer import (
    RoboDojoTrainer,
    joint_multiplier,
)


def test_history_and_padding_survive_native_collation_and_transfer():
    example = {
        "action": torch.arange(32 * 16).reshape(32, 16),
        "condition_video": torch.arange(27).reshape(3, 9, 1, 1),
        "condition_video_pad_mask": torch.tensor([False] * 8 + [True]),
        "video_pad_mask": torch.tensor([False, True]),
        "sample_idx": torch.tensor(17),
        "instruction": "fold clothes",
    }
    batch = RoboDojoPreprocessor()([example, example]).to(torch.device("cpu"))
    assert set(batch.to_sample()) == set(example)
    for name, value in example.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(batch.to_sample()[name], torch.stack([value, value]))
    assert batch.action.shape == (2, 32, 16)


def test_full_joint_meta_freeze_and_optimizer_coverage():
    model = build_model(RoboDojoModelConfig())
    model.train()
    assert all(p.is_meta for p in model.parameters())
    assert all(p.requires_grad for p in model.video_expert.parameters())
    assert all(p.requires_grad for p in model.action_expert.parameters())
    assert all(p.requires_grad for p in model.proprio_encoder.parameters())
    assert model.mot.training and model.action_expert.training and model.video_expert.training
    assert not model.training and not model.vae.training and not model.clip_encoder.training
    assert not any(p.requires_grad for p in model.vae.parameters())
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 18_423_683_920
    args = SimpleNamespace(lr_base=1e-4, adam_beta1=0.9, adam_beta2=0.95, adam_eps=1e-8, weight_decay=0.01)
    optimizer = RoboDojoTrainer._build_optimizer(SimpleNamespace(model=model, training_args=args))
    video, action = optimizer.param_groups
    assert video["lr"] == 5e-6 and action["lr"] == 1e-4
    assert [id(p) for p in video["params"]] == [id(p) for p in model.video_expert.parameters()]
    flattened = [p for group in optimizer.param_groups for p in group["params"]]
    assert len({id(p) for p in flattened}) == len(flattened)
    assert {id(p) for p in flattened} == {id(p) for p in model.parameters() if p.requires_grad}


@pytest.mark.parametrize(
    "step,expected",
    [(0, 0.002), (499, 1.0), (500, 1.0), (80000, 0.01), (90000, 0.0055), (100000, 0.001), (100001, 0.001)],
)
def test_joint_schedule_preserves_released_timescale(step, expected):
    assert joint_multiplier(step) == pytest.approx(expected, abs=1e-14)
