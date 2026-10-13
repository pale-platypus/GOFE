#!/bin/bash
# Run B-MAFE(MI) at increasing operator-vocabulary sizes |O| on a single dataset
# (comment 5: vary the number of operators). Companion to run_maxorder_ablation.sh.
# Each run lands in outputs_num/nfo/RF/lsbo/mi_ops<label>/lam<λ>/<dataset>/ with
# a fresh pool/ckp dir (caches are vocabulary-specific).
#
# Usage:
#   PYTHON=... bash scripts/run_ops_ablation.sh <dataset> <cuda> [total_iter] [lam] [seed]
# Defaults: total_iter=3, lam=0.5 (matches the depth ablation protocol), seed=0.
# seed>0 runs land in ops_ablation/<dataset>_s<seed>/ with their own pool dirs,
# so multiple seeds can be aggregated by scripts/ops_ablation_table.py.

set -o pipefail

dataset=${1:?missing dataset}
cuda=${2:?missing cuda id}
total_iter=${3:-3}
lam=${4:-0.5}
seed=${5:-0}

SFX=""
if [ "$seed" != "0" ]; then SFX="_s${seed}"; fi

PYTHON=${PYTHON:-python}
export PYTHONIOENCODING=utf-8

# |O| ladder: arithmetic pair -> paper's binary core -> +2 unary -> full 9.
LABELS=(2 4 6 9)
OPS_LIST=("+,*" "+,-,*,division" "+,-,*,division,log,sqrt_abs" "")

OUT_ROOT="outputs_num"
ABL_DIR="$OUT_ROOT/quality_compare/ops_ablation/${dataset}${SFX}"
mkdir -p "$ABL_DIR"
LOG="$ABL_DIR/sweep.log"

{
    echo "================================================================="
    echo "Operator-vocabulary ablation: B-MAFE(MI) at varying |O|"
    echo "  dataset    = $dataset"
    echo "  cuda       = $cuda"
    echo "  total_iter = $total_iter"
    echo "  lam        = $lam"
    echo "  seed       = $seed"
    echo "  |O| ladder = ${LABELS[*]}"
    echo "  python     = $PYTHON"
    echo "================================================================="
} | tee "$LOG"

for i in "${!LABELS[@]}"; do
    N="${LABELS[$i]}"
    OPS="${OPS_LIST[$i]}"
    echo "" | tee -a "$LOG"
    echo "##### |O| = $N (ops='${OPS:-full}') #####" | tee -a "$LOG"
    POOL_DIR="$OUT_ROOT/nfo/RF/lsbo/mi_ops${N}${SFX}/lam${lam}/$dataset"
    CKP_DIR="$OUT_ROOT/nfo/RF/lsbo/mi_ops${N}${SFX}/lam${lam}/ckps_ops${N}${SFX}/$dataset"
    mkdir -p "$POOL_DIR" "$(dirname "$CKP_DIR")"

    EXTRA=()
    if [ -n "$OPS" ]; then EXTRA+=("--ops=$OPS"); fi

    t0=$(date +%s)
    if PYTHONPATH=. "$PYTHON" autolearn/feat_selection/nfo/iter_train.py \
            --eval_model=RF --data="$dataset" --cuda="$cuda" \
            --hyper_config=default --method=lsbo --seed="$seed" \
            --total_iter="$total_iter" --score_mode=mi --lam="$lam" \
            --feat_pool="$POOL_DIR" --ckp_path="$CKP_DIR" \
            --num_ops_only "${EXTRA[@]}" 2>&1 | tee -a "$LOG"; then
        :
    else
        echo "[ops] |O|=$N FAILED" | tee -a "$LOG"
        continue
    fi
    elapsed=$(( $(date +%s) - t0 ))
    echo "[ops] |O|=$N done in ${elapsed}s" | tee -a "$LOG"
    echo "$N,$elapsed" >> "$ABL_DIR/timing.csv"
done

# Header for timing.csv (idempotent — only add if missing)
if ! grep -q '^n_ops,seconds$' "$ABL_DIR/timing.csv" 2>/dev/null; then
    sort -u "$ABL_DIR/timing.csv" -o "$ABL_DIR/timing.csv"
    sed -i '1i n_ops,seconds' "$ABL_DIR/timing.csv"
fi

# Evaluate quality across all vocabulary sizes
echo "" | tee -a "$LOG"
echo "##### eval_quality across |O| #####" | tee -a "$LOG"
inputs=()
for N in "${LABELS[@]}"; do
    src="$OUT_ROOT/nfo/RF/lsbo/mi_ops${N}${SFX}/lam${lam}/$dataset/selected_topk_lam${lam}.json"
    if [ -f "$src" ]; then
        target="$ABL_DIR/selected_topk_ops${N}_lam${lam}.json"
        "$PYTHON" - <<PY
import json
with open("$src") as f:
    m = json.load(f)
m['method'] = f"lsbo_ops{$N}"
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
echo "Ops ablation done. Outputs:" | tee -a "$LOG"
echo "  $ABL_DIR/tables.csv      — per-|O| metrics" | tee -a "$LOG"
echo "  $ABL_DIR/timing.csv      — per-|O| wall-clock" | tee -a "$LOG"
echo "  $ABL_DIR/per_feature.csv — per-feature metrics (for plotting)" | tee -a "$LOG"
