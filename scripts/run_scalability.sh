#!/bin/bash
# Scalability sweep — 2D grid: dataset size N × B-MAFE training budget.
# OpenFE and AutoFeat are run once per N (no budget axis); their timings are
# appended to the same timing.csv so plot_scalability.py can overlay all three
# series.
#
# Idempotent: cells that already have an entry in timing.csv for the matching
# (method, n_rows, budget) tuple are skipped, so re-runs only fill in the gaps.
#
# Assumes scripts/make_subsamples.py was run first so data/<base>_n<N>.csv
# exists for each N.
#
# Usage:
#   PYTHON=... bash scripts/run_scalability.sh <base_dataset> <cuda> [total_iter]
#   SIZES="..." BUDGETS="..." PYTHON=... bash ... <base> <cuda> [iter]
# Default total_iter=3

set -o pipefail

base=${1:?missing base dataset name (e.g. amazon)}
cuda=${2:?missing cuda id}
total_iter=${3:-3}

PYTHON=${PYTHON:-python}
if [ -n "${SIZES:-}" ]; then
    # shellcheck disable=SC2206
    SIZES=(${SIZES})
else
    SIZES=(1000 4000 8000)        # N: dataset row counts
fi
if [ -n "${BUDGETS:-}" ]; then
    # shellcheck disable=SC2206
    BUDGETS=(${BUDGETS})
else
    BUDGETS=(256 1024)            # random_set_size grid for B-MAFE
fi
OUT_ROOT="outputs_num"
SCALE_DIR="$OUT_ROOT/quality_compare/scalability/$base"
mkdir -p "$SCALE_DIR"
LOG="$SCALE_DIR/sweep.log"

{
    echo "================================================================="
    echo "Scalability sweep (2D: N x random_set_size):"
    echo "  base       = $base"
    echo "  sizes (N)  = ${SIZES[*]}"
    echo "  budgets    = ${BUDGETS[*]}"
    echo "  total_iter = $total_iter (B-MAFE)"
    echo "  cuda       = $cuda"
    echo "  python     = $PYTHON"
    echo "================================================================="
} | tee -a "$LOG"

# Subsample sanity
for N in "${SIZES[@]}"; do
    if [ ! -f "data/${base}_n${N}.csv" ]; then
        echo "[scale] missing data/${base}_n${N}.csv — run scripts/make_subsamples.py first" | tee -a "$LOG"
        exit 2
    fi
done

# header
if [ ! -s "$SCALE_DIR/timing.csv" ]; then
    echo "method,n_rows,budget,seconds" > "$SCALE_DIR/timing.csv"
fi

# Returns 0 (success / "already done") if a row already exists matching
# method,n,budget (where budget="" for OpenFE/AutoFeat).
already_done() {
    local method=$1 n=$2 budget=$3
    [ -f "$SCALE_DIR/timing.csv" ] || return 1
    if [ -z "$budget" ]; then
        grep -q "^${method},${n},," "$SCALE_DIR/timing.csv"
    else
        grep -q "^${method},${n},${budget}," "$SCALE_DIR/timing.csv"
    fi
}

for N in "${SIZES[@]}"; do
    sub="${base}_n${N}"
    echo "" | tee -a "$LOG"
    echo "=================================================================" | tee -a "$LOG"
    echo "##### N = $N (dataset = $sub) #####" | tee -a "$LOG"

    # ---- B-MAFE(MI) at each budget
    for B in "${BUDGETS[@]}"; do
        if already_done "B-MAFE(MI)" "$N" "$B"; then
            echo "[scale] SKIP B-MAFE(MI) N=$N B=$B (already in timing.csv)" | tee -a "$LOG"
            continue
        fi
        echo "" | tee -a "$LOG"
        echo "----- B-MAFE(MI) budget=$B -----" | tee -a "$LOG"
        POOL_DIR="$OUT_ROOT/nfo/RF/lsbo/mi_scale_b${B}/lam0.5/$sub"
        CKP_DIR="$OUT_ROOT/nfo/RF/lsbo/mi_scale_b${B}/lam0.5/ckps_scale/$sub"
        mkdir -p "$POOL_DIR" "$(dirname "$CKP_DIR")"
        t0=$(date +%s)
        if PYTHONPATH=. "$PYTHON" autolearn/feat_selection/nfo/iter_train.py \
                --eval_model=RF --data="$sub" --cuda="$cuda" \
                --hyper_config=default --method=lsbo \
                --total_iter="$total_iter" --score_mode=mi --lam=0.5 \
                --random_set_size="$B" \
                --feat_pool="$POOL_DIR" --ckp_path="$CKP_DIR" \
                --num_ops_only 2>&1 | tee -a "$LOG"; then
            elapsed=$(( $(date +%s) - t0 ))
            echo "B-MAFE(MI),$N,$B,$elapsed" >> "$SCALE_DIR/timing.csv"
            echo "[scale] B-MAFE(MI) N=$N B=$B done in ${elapsed}s" | tee -a "$LOG"
        else
            echo "[scale] B-MAFE(MI) N=$N B=$B FAILED" | tee -a "$LOG"
        fi
    done

    # ---- OpenFE — no budget axis; one timing per N
    if already_done "OpenFE" "$N" ""; then
        echo "[scale] SKIP OpenFE N=$N (already in timing.csv)" | tee -a "$LOG"
    else
        OFE_DIR="$OUT_ROOT/nfo/RF/openfe/iter_scale/$sub"
        mkdir -p "$OFE_DIR"
        echo "" | tee -a "$LOG"
        echo "----- OpenFE -----" | tee -a "$LOG"
        t0=$(date +%s)
        if PYTHONPATH=. "$PYTHON" autolearn/feat_selection/openfe_baseline.py \
                --data="$sub" --eval_model=RF --n_jobs=4 \
                --feat_pool="$OFE_DIR" 2>&1 | tee -a "$LOG"; then
            elapsed=$(( $(date +%s) - t0 ))
            echo "OpenFE,$N,,$elapsed" >> "$SCALE_DIR/timing.csv"
            echo "[scale] OpenFE N=$N done in ${elapsed}s" | tee -a "$LOG"
        else
            echo "[scale] OpenFE N=$N FAILED" | tee -a "$LOG"
        fi
    fi

    # ---- AutoFeat — no budget axis; one timing per N
    if already_done "AutoFeat" "$N" ""; then
        echo "[scale] SKIP AutoFeat N=$N (already in timing.csv)" | tee -a "$LOG"
    else
        AF_DIR="$OUT_ROOT/nfo/RF/autofeat/iter_scale/$sub"
        mkdir -p "$AF_DIR"
        echo "" | tee -a "$LOG"
        echo "----- AutoFeat -----" | tee -a "$LOG"
        t0=$(date +%s)
        if PYTHONPATH=. "$PYTHON" autolearn/feat_selection/autofeat_baseline.py \
                --data="$sub" --eval_model=RF --n_jobs=4 \
                --feat_pool="$AF_DIR" 2>&1 | tee -a "$LOG"; then
            elapsed=$(( $(date +%s) - t0 ))
            echo "AutoFeat,$N,,$elapsed" >> "$SCALE_DIR/timing.csv"
            echo "[scale] AutoFeat N=$N done in ${elapsed}s" | tee -a "$LOG"
        else
            echo "[scale] AutoFeat N=$N FAILED" | tee -a "$LOG"
        fi
    fi
done

echo "" | tee -a "$LOG"
echo "Scalability sweep done." | tee -a "$LOG"
echo "  timing.csv: $SCALE_DIR/timing.csv" | tee -a "$LOG"
