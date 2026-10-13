#!/bin/bash
# Run B-MAFE(MI) at multiple maximum expression depths (Dmax) on a single dataset.
# Each run lands in outputs_num/nfo/RF/lsbo/mi_depth<D>/lam<λ>/<dataset>/
# so they don't collide with the standard sweep or the budget ablation.
#
# Usage:
#   PYTHON=... bash scripts/run_maxorder_ablation.sh <dataset> <cuda> [total_iter] [lam]
# Defaults: total_iter=3, lam=0.5

set -o pipefail

dataset=${1:?missing dataset}
cuda=${2:?missing cuda id}
total_iter=${3:-3}
lam=${4:-0.5}

PYTHON=${PYTHON:-python}
DEPTHS=(2 3 4 5)

OUT_ROOT="outputs_num"
ABL_DIR="$OUT_ROOT/quality_compare/depth_ablation/$dataset"
mkdir -p "$ABL_DIR"
LOG="$ABL_DIR/sweep.log"

{
    echo "================================================================="
    echo "Depth ablation: B-MAFE(MI) at varying max_order (Dmax)"
    echo "  dataset    = $dataset"
    echo "  cuda       = $cuda"
    echo "  total_iter = $total_iter"
    echo "  lam        = $lam"
    echo "  depths     = ${DEPTHS[*]}"
    echo "  python     = $PYTHON"
    echo "================================================================="
} | tee "$LOG"

for D in "${DEPTHS[@]}"; do
    echo "" | tee -a "$LOG"
    echo "##### max_order = $D #####" | tee -a "$LOG"
    POOL_DIR="$OUT_ROOT/nfo/RF/lsbo/mi_depth${D}/lam${lam}/$dataset"
    CKP_DIR="$OUT_ROOT/nfo/RF/lsbo/mi_depth${D}/lam${lam}/ckps_depth${D}/$dataset"
    mkdir -p "$POOL_DIR" "$(dirname "$CKP_DIR")"

    t0=$(date +%s)
    if PYTHONPATH=. "$PYTHON" autolearn/feat_selection/nfo/iter_train.py \
            --eval_model=RF --data="$dataset" --cuda="$cuda" \
            --hyper_config=default --method=lsbo \
            --total_iter="$total_iter" --score_mode=mi --lam="$lam" \
            --max_order="$D" \
            --feat_pool="$POOL_DIR" --ckp_path="$CKP_DIR" \
            --num_ops_only 2>&1 | tee -a "$LOG"; then
        :
    else
        echo "[depth] D=$D FAILED" | tee -a "$LOG"
        continue
    fi
    elapsed=$(( $(date +%s) - t0 ))
    echo "[depth] D=$D done in ${elapsed}s" | tee -a "$LOG"
    echo "$D,$elapsed" >> "$ABL_DIR/timing.csv"
done

# Header for timing.csv (idempotent — only add if missing)
if ! grep -q '^depth,seconds$' "$ABL_DIR/timing.csv" 2>/dev/null; then
    sort -u "$ABL_DIR/timing.csv" -o "$ABL_DIR/timing.csv"
    sed -i '1i depth,seconds' "$ABL_DIR/timing.csv"
fi

# Evaluate quality across all depths
echo "" | tee -a "$LOG"
echo "##### eval_quality across depths #####" | tee -a "$LOG"
inputs=()
for D in "${DEPTHS[@]}"; do
    src="$OUT_ROOT/nfo/RF/lsbo/mi_depth${D}/lam${lam}/$dataset/selected_topk_lam${lam}.json"
    if [ -f "$src" ]; then
        target="$ABL_DIR/selected_topk_depth${D}_lam${lam}.json"
        "$PYTHON" - <<PY
import json
with open("$src") as f:
    m = json.load(f)
m['method'] = f"lsbo_depth{$D}"
with open("$target", "w") as f:
    json.dump(m, f, indent=2)
PY
        inputs+=("$target")
    else
        echo "[eval] WARNING: missing $src" | tee -a "$LOG"
    fi
done

if [ ${#inputs[@]} -gt 0 ]; then
    PYTHONPATH=. "$PYTHON" -m autolearn.feat_selection.eval_quality \
        --data "$dataset" \
        --inputs "${inputs[@]}" \
        --output "$ABL_DIR/tables.csv" \
        --checkpoint_dir "$ABL_DIR" \
        --include_base_features \
        --lambdas 0.0 0.5 1.0 \
        --top_k 16 \
        2>&1 | tee "$ABL_DIR/eval.log"
fi

echo "" | tee -a "$LOG"
echo "Depth ablation done. Outputs:" | tee -a "$LOG"
echo "  $ABL_DIR/tables.csv      — per-depth metrics" | tee -a "$LOG"
echo "  $ABL_DIR/timing.csv      — per-depth wall-clock" | tee -a "$LOG"
echo "  $ABL_DIR/per_feature.csv — per-feature metrics (for plotting)" | tee -a "$LOG"
