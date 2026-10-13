#!/bin/bash
# Run B-MAFE(MI) at multiple random_set_size budgets on a single dataset.
# Each run lands in outputs_num/nfo/RF/lsbo/mi_budget<N>/lam<λ>/<dataset>/
# so they don't collide with the standard sweep.
#
# Usage:
#   PYTHON=... bash scripts/run_budget_ablation.sh <dataset> <cuda> [total_iter] [lam]
# Defaults: total_iter=3, lam=0.5

set -o pipefail

dataset=${1:?missing dataset}
cuda=${2:?missing cuda id}
total_iter=${3:-3}
lam=${4:-0.5}

PYTHON=${PYTHON:-python}
BUDGETS=(128 256 512 1024 2048)

OUT_ROOT="outputs_num"
ABL_DIR="$OUT_ROOT/quality_compare/budget_ablation/$dataset"
mkdir -p "$ABL_DIR"
LOG="$ABL_DIR/sweep.log"

{
    echo "================================================================="
    echo "Budget ablation: B-MAFE(MI) at varying random_set_size"
    echo "  dataset    = $dataset"
    echo "  cuda       = $cuda"
    echo "  total_iter = $total_iter"
    echo "  lam        = $lam"
    echo "  budgets    = ${BUDGETS[*]}"
    echo "  python     = $PYTHON"
    echo "================================================================="
} | tee "$LOG"

for B in "${BUDGETS[@]}"; do
    echo "" | tee -a "$LOG"
    echo "##### random_set_size = $B #####" | tee -a "$LOG"
    POOL_DIR="$OUT_ROOT/nfo/RF/lsbo/mi_budget${B}/lam${lam}/$dataset"
    CKP_DIR="$OUT_ROOT/nfo/RF/lsbo/mi_budget${B}/lam${lam}/ckps_budget${B}/$dataset"
    mkdir -p "$POOL_DIR" "$(dirname "$CKP_DIR")"

    t0=$(date +%s)
    if PYTHONPATH=. "$PYTHON" autolearn/feat_selection/nfo/iter_train.py \
            --eval_model=RF --data="$dataset" --cuda="$cuda" \
            --hyper_config=default --method=lsbo \
            --total_iter="$total_iter" --score_mode=mi --lam="$lam" \
            --random_set_size="$B" \
            --feat_pool="$POOL_DIR" --ckp_path="$CKP_DIR" \
            --num_ops_only 2>&1 | tee -a "$LOG"; then
        :
    else
        echo "[budget] B=$B FAILED" | tee -a "$LOG"
        continue
    fi
    elapsed=$(( $(date +%s) - t0 ))
    echo "[budget] B=$B done in ${elapsed}s" | tee -a "$LOG"
    # Persist wall-clock per budget for the plotter
    echo "$B,$elapsed" >> "$ABL_DIR/timing.csv"
done

# Header for timing.csv (idempotent — only add if missing)
if ! grep -q '^budget,seconds$' "$ABL_DIR/timing.csv" 2>/dev/null; then
    sort -u "$ABL_DIR/timing.csv" -o "$ABL_DIR/timing.csv"
    sed -i '1i budget,seconds' "$ABL_DIR/timing.csv"
fi

# Evaluate quality across all budgets
echo "" | tee -a "$LOG"
echo "##### eval_quality across budgets #####" | tee -a "$LOG"
inputs=()
for B in "${BUDGETS[@]}"; do
    src="$OUT_ROOT/nfo/RF/lsbo/mi_budget${B}/lam${lam}/$dataset/selected_topk_lam${lam}.json"
    if [ -f "$src" ]; then
        # Tag each manifest with its budget by copying to per-budget eval-manifest
        # and re-labeling the method so it appears as a distinct row in tables.csv.
        target="$ABL_DIR/selected_topk_budget${B}_lam${lam}.json"
        "$PYTHON" - <<PY
import json
with open("$src") as f:
    m = json.load(f)
m['method'] = f"lsbo_budget{$B}"
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
echo "Budget ablation done. Outputs:" | tee -a "$LOG"
echo "  $ABL_DIR/tables.csv      — per-budget metrics" | tee -a "$LOG"
echo "  $ABL_DIR/timing.csv      — per-budget wall-clock" | tee -a "$LOG"
echo "  $ABL_DIR/per_feature.csv — per-feature metrics (for plotting)" | tee -a "$LOG"
