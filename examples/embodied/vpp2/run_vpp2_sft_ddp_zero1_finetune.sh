#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOONGFORGE_PATH="${LOONGFORGE_PATH:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
exec "$SCRIPT_DIR/run_vpp2_sft_ddp_finetune.sh" --zero-optimizer --zero-master-param-dtype fp32 "$@"
