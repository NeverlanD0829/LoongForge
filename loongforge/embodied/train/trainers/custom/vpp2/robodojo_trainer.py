# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Native trainer with the released RoboDojo parameter groups and LR schedule."""

import math

import torch

from loongforge.embodied.distributed.utils import unwrap_model
from loongforge.embodied.train.trainers.supervised.finetune_trainer import FinetuneTrainer


def joint_multiplier(step: int, warmup: int = 500) -> float:
    """Preserve the released warmup, 80k cosine decay and 100k tail."""
    step = max(int(step), 0)
    if step > 80000:
        progress = min(max((step - 80000) / 20000, 0.0), 1.0)
        return 0.001 + (0.01 - 0.001) * (1 + math.cos(math.pi * progress)) / 2
    if warmup > 0 and step < warmup:
        return float(step + 1) / float(warmup)
    progress = (step - warmup) / (80000 - warmup)
    return 0.01 + (1.0 - 0.01) * 0.5 * (1.0 + math.cos(math.pi * progress))


class RoboDojoTrainer(FinetuneTrainer):
    def _build_optimizer(self):
        model = unwrap_model(self.model)
        video = list(dict.fromkeys(model.video_expert.parameters()))
        action = list(dict.fromkeys([*model.action_expert.parameters(), *model.proprio_encoder.parameters()]))
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        if {id(p) for p in video + action} != expected or {id(p) for p in video} & {id(p) for p in action}:
            raise RuntimeError("Joint parameter groups must partition every trainable parameter")
        args = self.training_args
        return torch.optim.AdamW(
            [
                {"params": video, "lr": 5e-6},
                {"params": action, "lr": 1e-4},
            ],
            lr=args.lr_base,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )

    def _build_scheduler(self):
        warmup = self.training_args.lr_warmup_iters
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: joint_multiplier(step, warmup))
