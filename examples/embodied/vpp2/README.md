# VPP2 training in LoongForge

VPP2 provides two native embodied training entries:

| Model | Dataset | Trainable components | Launcher |
|---|---|---|---|
| `vpp2` | LIBERO | Action DiT and proprio encoder; Video DiT frozen | `run_vpp2_sft_deepspeed_zero2_finetune.sh` |
| `vpp2_robodojo` | RoboDojo EE16 | Full Video DiT, Action2B and proprio encoder | `run_robodojo_joint_50.sh` |

Both use LoongForge's model, dataset, sampler and trainer registries. VAE,
CLIP and text encoders stay frozen. Run the scripts with the image's system
Python (`/usr/bin/python3` by default); `LOONGFORGE_PYTHON_BIN` overrides it.
The community implementation uses its own environment when collecting a
reference. Do not activate that community virtualenv for LoongForge.

## LIBERO assets and launch

Set `VPP2_ASSETS_ROOT` to the prepared asset directory. The default is
`vpp2_assets/` under the LoongForge repository. It must contain:

- `Wan2.1-I2V-14B-480P/`: shared Wan VAE and CLIP weights.
- `libero_video_step010000.pt`: Video-10k initialization. Override
  `VPP2_VIDEO_CHECKPOINT` when this file lives elsewhere.
- `libero/action_init.pt` and `libero/dataset_stats.json`: action backbone
  initialization and normalization statistics.
- `libero_subset/`: the four LIBERO LeRobot suites. Override
  `VPP2_DATASET_ROOT` or the comma-separated `DATASET_PATH` if needed.
- `text_cache_real/`: precomputed text embeddings, using cache tag
  `wan21i2v14b480p` and the released 128-token context.

```bash
cd /path/to/LoongForge
export VPP2_ASSETS_ROOT=/path/to/vpp2_assets

OUTPUT_DIR=/path/to/new_libero_run \
  bash examples/embodied/vpp2/run_vpp2_sft_deepspeed_zero2_finetune.sh
```

The ZeRO-2 defaults are 8 GPUs, batch 2 per device, accumulation 1, 50 updates,
seed 42 and 16 workers. AdamW uses betas `(0.9, 0.95)`, weight decay `0.01`,
LR `1e-4`, two warmup updates and cosine decay to `1e-6`. Set `SCHEDULE_STEPS`
to keep a longer learning-rate schedule when stopping early. Checkpoint saves
are disabled by default. The separate `run_vpp2_sft_ddp_finetune.sh` retains
the DDP recipe and has its own batch, schedule and save defaults.

## RoboDojo assets and launch

Prepare the public 3500-episode EE16 export using the community
[RoboDojo training guide](https://github.com/Haodong082399/video-prediction-policy-2/blob/main/docs/robodojo.md).
The joint stage needs the fixed 3466/34 episode split, official EE16 statistics,
critical-phase index and all 35 prompt caches. Initialization uses the released
history-conditioned Video-10k and its matching Action2B initializer.

Create a resource environment file such as `/path/to/robodojo_resources.sh`:

```bash
export VPP2_PREPARED=/path/to/prepared/robodojo
export VPP2_MEDIA_ROOT=/path/to/native_frame_media
export VPP2_WAN_ROOT=/path/to/Wan2.1-I2V-14B-480P
export VPP2_VIDEO_INIT=/path/to/robodojo_his10k.pt
export VPP2_ACTION_INIT=/path/to/action2b_from_his10k.pt
```

```bash
cd /path/to/LoongForge
ROBODOJO_RESOURCES=/path/to/robodojo_resources.sh \
OUTPUT_DIR=/path/to/new_robodojo_run \
TRAIN_ITERS=50 \
  bash examples/embodied/vpp2/run_robodojo_joint_50.sh
```

On the prepared development machine, the resource file defaults to
`/workspace/vpp2_assets/robodojo_training_resources.sh`. The output defaults to
`outputs/vpp2_robodojo_zero2/<timestamp>` under the repository.

The recipe uses all 40 Video and 40 Action2B layers, EE16 actions, proprio16,
eight history frames and 18,423,683,920 trainable elements. Defaults are
8 GPUs × batch 2 × accumulation 1 = global batch 16, 50 updates, seed 100008
and four data workers. ZeRO-2 offloads FP32 AdamW master weights/state to CPU
to fit the full joint model on 8 × 80 GB GPUs.

`RoboDojoTrainer` owns the two parameter groups: Video LR `5e-6` and
Action/proprio LR `1e-4`. It preserves the original 500-step warmup, cosine
decay through 80k and tail through 100k; an early stop does not compress the
schedule. AdamW uses betas `(0.9, 0.95)`, weight decay `0.01` and gradient
clipping `1.0`. Evaluation and checkpoint saves are disabled for this short run.

The launcher uses strict deterministic algorithms, `OMP_NUM_THREADS=8` and
`MKL_NUM_THREADS=1`. Single-threaded MKL avoids a CPU AdamW sqrt crash observed
on this image. It explicitly selects the image's cuBLAS via `LD_PRELOAD`:
community and system Python can otherwise load different cuBLAS versions even
when using the same Torch binary. Set `ROBODOJO_CUBLAS_ROOT` for a different
image library location. Keep this runtime policy fixed when comparing runs.

## Overrides and checkpoint behavior

The launchers group model, data, training, distributed and logging arguments.
Training options and `model.*` / `data.*` overrides are forwarded to LoongForge:

```bash
TRAIN_ITERS=100 SEED=100008 \
  bash examples/embodied/vpp2/run_robodojo_joint_50.sh

bash examples/embodied/vpp2/run_robodojo_joint_50.sh \
  --train-iters 100 model.action_horizon=16
```

Common environment overrides are `OUTPUT_DIR`, `TRAIN_ITERS`,
`PER_DEVICE_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`, `NUM_WORKERS`,
`SAVE_INTERVAL`, `DATASET_PATH` and `DEEPSPEED_CONFIG`. Defaults above describe
the validated recipes; changing batch geometry, scheduler or runtime policy
requires fresh validation. System DeepSpeed versions below 0.18.9 support this
integration only with accumulation 1; the backend rejects larger accumulation.

Seeded eager construction preserves the community factory's initialization
order for projections excluded from the action backbone checkpoint.
`model.initialization_seed` selects that model seed, and starts its training
RNG stream after data-loader setup. Resume restores checkpoint RNG instead.

When enabled with `SAVE_INTERVAL`, native DeepSpeed saves model, optimizer,
scheduler, rank RNG and data-loader state. All ranks participate; resume
requires the original world size. Model loading remains strict: missing or
unexpected tensors are errors.

See the [model package](../../../loongforge/embodied/model/vpp2/README.md) for
architecture and source-license details. CPU regression tests live in
`tests/embodied/unit/test_vpp2*.py`. Historical alignment reports, captured
states and one-off diagnostic programs are maintained outside the source tree.
