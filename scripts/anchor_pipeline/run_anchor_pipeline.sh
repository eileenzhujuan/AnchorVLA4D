#!/usr/bin/env bash
# Run the full anchor-processing pipeline in order:
# 1) process_anchor.py  -> generate anchor metadata
# 2) add_anchor.py      -> inject anchors into the dataset
# 3) filter_bridge_images.py -> keep only one anchor per sample
#
# Usage:
#   bash run_anchor_pipeline.sh <input_json> [metadata_output_name] [with_anchor_output] [filtered_output]

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROCESS_SCRIPT="$SCRIPT_DIR/process_anchor.py"
ADD_SCRIPT="$SCRIPT_DIR/add_anchor.py"
FILTER_SCRIPT="$SCRIPT_DIR/filter_bridge_images.py"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <input_json> [metadata_output_name] [with_anchor_output] [filtered_output]"
  exit 1
fi

INPUT_JSON=$1
METADATA_OUTPUT_NAME=${2:-sematic_anchors.json}
WITH_ANCHOR_OUTPUT=${3:-}
FILTERED_OUTPUT=${4:-}

INPUT_ABS=$(python3 - <<'PY' "$INPUT_JSON"
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)

INPUT_DIR=$(dirname "$INPUT_ABS")
INPUT_BASENAME=$(basename "$INPUT_ABS")
INPUT_STEM=${INPUT_BASENAME%.*}

METADATA_OUTPUT_PATH=$(python3 - <<'PY' "$INPUT_ABS" "$METADATA_OUTPUT_NAME"
import os, sys
json_file = os.path.abspath(sys.argv[1])
out_name = sys.argv[2]
data_root = os.path.dirname(os.path.dirname(json_file))
if not os.path.exists(data_root):
    data_root = '/root/datasets'
print(os.path.join(data_root, out_name))
PY
)

if [[ -z "$WITH_ANCHOR_OUTPUT" ]]; then
  WITH_ANCHOR_OUTPUT="$INPUT_DIR/${INPUT_STEM}_with_anchor.json"
fi

if [[ -z "$FILTERED_OUTPUT" ]]; then
  FILTERED_OUTPUT="$INPUT_DIR/${INPUT_STEM}_filtered.json"
fi

LOG_DIR="$INPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${INPUT_STEM}_anchor_pipeline.log"
: > "$LOG_FILE"

run_and_log() {
  local step=$1
  shift
  echo "[$step] $*" >> "$LOG_FILE"
  echo "[$step] Running: $*" >&2
  "$@" >> "$LOG_FILE" 2>&1
}

echo "[1/3] Running process_anchor.py" | tee -a "$LOG_FILE"
python3 "$PROCESS_SCRIPT" "$INPUT_ABS" "$METADATA_OUTPUT_NAME" 2>&1 | tee -a "$LOG_FILE"

echo "[2/3] Running add_anchor.py" | tee -a "$LOG_FILE"
python3 "$ADD_SCRIPT" "$INPUT_ABS" --metadata "$METADATA_OUTPUT_PATH" --output "$WITH_ANCHOR_OUTPUT" 2>&1 | tee -a "$LOG_FILE"

echo "[3/3] Running filter_bridge_images.py" | tee -a "$LOG_FILE"
python3 "$FILTER_SCRIPT" "$WITH_ANCHOR_OUTPUT" "$FILTERED_OUTPUT" 2>&1 | tee -a "$LOG_FILE"

echo "Pipeline completed." | tee -a "$LOG_FILE"
echo "Metadata: $METADATA_OUTPUT_PATH" | tee -a "$LOG_FILE"
echo "With anchors: $WITH_ANCHOR_OUTPUT" | tee -a "$LOG_FILE"
echo "Filtered: $FILTERED_OUTPUT" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE"
