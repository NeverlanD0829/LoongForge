# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""CPU differential tests against the live, explicitly selected BASE checkout.

Tiny towers exercise actual action-only MoT/loss/backward without claiming
14B BF16, real VAE/CLIP, ZeRO updates or benchmark precision acceptance.
"""

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from loongforge.embodied.data.datasets.sampler_builder import SamplerBuilderContext
from loongforge.embodied.data.datasets.vpp2.samplers import build_vpp2_sampler
from loongforge.embodied.model.vpp2.modeling_configuration_vpp2 import VPP2ModelConfig
from loongforge.embodied.model.vpp2.modeling_vpp2 import VPP2Model


def test_single_rank_sampler_does_not_consume_global_rng():
    context = SamplerBuilderContext(
        range(579),
        SimpleNamespace(batch_drop_last=False),
        SimpleNamespace(world_size=1, rank=0, is_distributed=False),
        batch_size=32,
        seed=42,
        shuffle=True,
    )
    sampler = build_vpp2_sampler(context)
    torch.manual_seed(5)
    before = torch.get_rng_state()
    actual = list(sampler)
    assert torch.equal(before, torch.get_rng_state())
    expected = torch.randperm(579, generator=torch.Generator().manual_seed(42)).tolist()
    assert actual == expected


def test_eager_factory_preserves_community_initialization_and_seed(monkeypatch):
    import inspect

    signature = inspect.signature(VPP2Model.from_wan21_14b_pretrained)
    observed = []

    class Loaded(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(8, 16).to(torch.bfloat16)

        def freeze_modules(self):
            self.frozen = True

    def factory(cls, **kwargs):
        signature.bind(**kwargs)
        observed.append(kwargs)
        return Loaded()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(VPP2Model, "from_wan21_14b_pretrained", classmethod(factory))
    original = torch.get_rng_state()
    try:
        cfg = VPP2ModelConfig(init_on_meta=False, initialization_seed=42)
        first = VPP2Model.from_pretrained(cfg)
        torch.rand(51)
        second = VPP2Model.from_pretrained(cfg)
        assert torch.equal(first.projection.weight, second.projection.weight)
        torch.manual_seed(42)
        expected = torch.nn.Linear(8, 16).to(torch.bfloat16)
        assert torch.equal(first.projection.weight, expected.weight)
        assert first.config is cfg and first.frozen
        assert observed[0]["torch_dtype"] == torch.bfloat16
        assert observed[0]["video_train_beta_alpha"] == cfg.video_scheduler["beta_alpha"]
    finally:
        torch.set_rng_state(original)


def test_initialization_seed_never_overwrites_resume_rng():
    model = SimpleNamespace(config=VPP2ModelConfig(initialization_seed=42))
    before = torch.get_rng_state()
    VPP2Model.on_after_data_iterators_initialized(model, args=SimpleNamespace(resume=True), ctx=SimpleNamespace(rank=0))
    assert torch.equal(before, torch.get_rng_state())


@pytest.fixture
def base_modules():
    root = os.environ.get("VPP2_BASE_ROOT")
    if not root:
        pytest.skip("Set VPP2_BASE_ROOT for live differential tests")
    sys.path.insert(0, str(Path(root) / "src"))
    return {
        name: importlib.import_module("vpp2." + path)
        for name, path in {
            "video": "models.wan21_14b.wan_video_dit",
            "action": "models.wan21_14b.action_dit",
            "mot": "models.wan21_14b.mot",
            "model": "libero.action_model",
            "trainer": "libero.trainer",
        }.items()
    }


class _TinyVAE(torch.nn.Module):
    temporal_downsample_factor = 4

    def encode(self, videos, **kwargs):
        return [
            torch.nn.functional.adaptive_avg_pool3d(v.mean(0, keepdim=True), (1, 4, 4)).repeat(16, 1, 1, 1)
            for v in videos
        ]


def _tiny_model(mods):
    video = mods["video"].WanVideoDiT(
        dim=24,
        hidden_dim=24,
        ffn_dim=48,
        freq_dim=8,
        text_dim=16,
        num_heads=2,
        attn_head_dim=12,
        num_layers=2,
        use_gradient_checkpointing=False,
    )
    action = mods["action"].ActionDiT(
        hidden_dim=16,
        action_dim=7,
        ffn_dim=32,
        text_dim=16,
        freq_dim=8,
        eps=1e-6,
        num_heads=2,
        attn_head_dim=12,
        num_layers=2,
    )
    mot = mods["mot"].MoT({"video": video, "action": action}, mot_checkpoint_mixed_attn=False)
    model = mods["model"].LiberoActionModel(
        video_expert=video,
        action_expert=action,
        mot=mot,
        vae=_TinyVAE(),
        text_dim=16,
        proprio_dim=8,
        device="cpu",
        torch_dtype=torch.float32,
        video_train_sampling_strategy="pure_noise",
        action_visible_video_frames=3,
        action_only_first_frame_fast_path=True,
    )
    model.eval().requires_grad_(False)
    model.action_expert.train().requires_grad_(True)
    model.proprio_encoder.train().requires_grad_(True)
    return model


def test_live_action_loss_gradient_and_adam_updates_are_exact(base_modules):
    prefix = "loongforge.embodied.model.vpp2.modules."
    trans_modules = {
        name: importlib.import_module(prefix + path)
        for name, path in {
            "video": "wan_video_dit",
            "action": "action_dit",
            "mot": "mot",
            "model": "libero_action_model",
        }.items()
    }
    base, trans = _tiny_model(base_modules), _tiny_model(trans_modules)
    trans.load_state_dict(base.state_dict(), strict=True)
    torch.manual_seed(123)
    sample = dict(
        video=torch.randn(2, 3, 17, 32, 32),
        action=torch.randn(2, 32, 7),
        proprio=torch.randn(2, 1, 8),
        context=torch.randn(2, 4, 16),
        context_mask=torch.ones(2, 4, dtype=torch.bool),
        image_is_pad=torch.zeros(2, 17, dtype=torch.bool),
        action_is_pad=torch.zeros(2, 32, dtype=torch.bool),
        clip_fea=torch.randn(2, 257, 1280),
    )
    sample["action_is_pad"][1, 27:] = True
    opts = [
        torch.optim.AdamW(
            [p for p in m.parameters() if p.requires_grad], lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01, foreach=False
        )
        for m in (base, trans)
    ]
    for step in range(3):
        results = []
        for model, opt in zip((base, trans), opts, strict=True):
            torch.manual_seed(42 + step)
            opt.zero_grad(set_to_none=True)
            loss, info = model.training_loss(sample, training_stage="action")
            loss.backward()
            assert all(p.grad is None for p in model.video_expert.parameters())
            results.append(
                (loss.detach(), info, {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad})
            )
        assert torch.equal(results[0][0], results[1][0])
        assert results[0][1] == results[1][1]
        assert results[0][2].keys() == results[1][2].keys()
        for name, grad in results[0][2].items():
            assert torch.equal(grad, results[1][2][name]), name
        for opt in opts:
            opt.step()
        for (name, p), (_, q) in zip(base.named_parameters(), trans.named_parameters(), strict=True):
            assert torch.equal(p, q), name


def test_full_lr_trace_matches_live_libero(base_modules):
    from loongforge.embodied.optimizer.lr_scheduler import build_scheduler

    for total, warmup in [(30000, 500), (100, 500), (10, 0)]:
        optimizers = [torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-4) for _ in range(2)]
        trainer = object.__new__(base_modules["trainer"].LiberoTrainer)
        trainer.optimizer = optimizers[0]
        trainer.learning_rate = 1e-4
        trainer.lr_min_ratio = 0.01
        trainer.lr_tail = trainer.lr_continuation = None
        schedulers = [
            trainer._build_scheduler("cosine", total, warmup),
            build_scheduler(
                optimizers[1],
                SimpleNamespace(
                    lr_decay_style="linear_warmup_cosine_annealing",
                    lr_decay_iters=total,
                    train_iters=100,
                    lr_warmup_iters=warmup,
                    min_lr=1e-6,
                ),
            ),
        ]
        for step in range(total + 1):
            # BASE computes lr*0.01; CLI stores 1e-6 (one fp64 ULP apart).
            assert optimizers[0].param_groups[0]["lr"] == pytest.approx(
                optimizers[1].param_groups[0]["lr"], rel=1e-12, abs=1e-18
            ), step
            for opt, scheduler in zip(optimizers, schedulers, strict=True):
                opt.step()
                scheduler.step()


def test_native_vpp2_has_no_chunk_contract_and_owns_optimizer_policy():
    assert not hasattr(VPP2Model, "split_training_batch")
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
    model = SimpleNamespace(config=VPP2ModelConfig())
    VPP2Model.on_after_data_iterators_initialized(
        model, args=SimpleNamespace(optimizer="AdamW", resume=False),
        ctx=SimpleNamespace(rank=0), optimizer=optimizer,
    )
    assert optimizer.param_groups[0]["foreach"] is None


def test_action_noise_timesteps_preserve_action_dtype():
    from loongforge.embodied.model.vpp2.modules.libero_action_model import LiberoActionModel

    action = torch.zeros(2, 32, 7, dtype=torch.bfloat16)
    scheduler = SimpleNamespace(
        sample_training_t=lambda *, batch_size, device, dtype: torch.full((batch_size,), 999.0, dtype=dtype),
        add_noise=lambda value, noise, timestep: value,
        training_target=lambda value, noise, timestep: noise,
    )
    inputs = dict(
        input_latents=torch.zeros(2, 16, 5, 2, 2, dtype=torch.bfloat16),
        condition_latents=torch.zeros(2, 16, 1, 2, 2, dtype=torch.bfloat16),
        context=None, context_mask=None, action=action, action_is_pad=None,
        image_is_pad=None, clip_fea=None, action_only_first_frame_fast_path=False,
        fuse_vae_embedding_in_latents=False,
    )
    model = SimpleNamespace(
        build_inputs=lambda *args, **kwargs: inputs,
        device="cpu", train_video_scheduler=SimpleNamespace(num_train_timesteps=1000),
        train_action_scheduler=scheduler,
        _training_loss_action_only=lambda **kwargs: kwargs["timestep_action"],
    )
    observed = LiberoActionModel.training_loss(model, {})
    assert observed.dtype == action.dtype
    # BF16's endpoint rounding is part of BASE's current numerical policy.
    assert observed.tolist() == [1000.0, 1000.0]
