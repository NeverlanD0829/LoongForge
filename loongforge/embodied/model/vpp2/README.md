# VPP2 in LoongForge embodied

This package integrates VPP2 through LoongForge's native model, dataset,
sampler and trainer registries:

- `vpp2`: LIBERO action-stage training, with frozen Wan Video DiT and trainable
  Action DiT/proprio encoder.
- `vpp2_robodojo`: RoboDojo joint Video + Action2B training, with EE16 actions,
  proprio16 and eight history frames.

The numerical modules in `modules/` retain the community implementation's
model and flow-matching behavior. The wrappers own framework configuration,
strict checkpoint loading, trainable/frozen module policy and loss reporting.
VAE, CLIP and text encoders stay frozen and in evaluation mode.

The RoboDojo data pipeline retains the native frame mapping, task balancing,
critical-phase sampling, history conditioning, normalization and padding
masks. Its community-derived implementation is under
`data/datasets/vpp2/robodojo_impl/`, with its source notice retained.
The sampler shards complete logical batches and supports data-loader resume.

The normal training path constructs and strictly loads the model eagerly
before distributed wrapping. Meta construction supports storage-free
architecture checks. Checkpoint-excluded
projections use the community factory's seeded initialization order.

For resource layout, launch commands and supported runtime defaults, see
[the training examples](../../../../examples/embodied/vpp2/README.md).
The image system environment is used for LoongForge; the community reference
keeps its separate environment. Model-specific numerical settings belong in
the VPP2 configs and launchers.

## Source and license

Adapted from
[`video-prediction-policy-2`](https://github.com/Haodong082399/video-prediction-policy-2).
Its extracted project code is MIT-licensed by the FastWAM Authors; Wan and
DiffSynth components retain their Apache-2.0 lineage. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the RoboDojo data pipeline's
[NOTICE.md](../../data/datasets/vpp2/robodojo_impl/NOTICE.md) for notices.
