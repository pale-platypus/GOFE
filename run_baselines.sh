#!/bin/bash
# Run all configured baselines on a dataset and produce per-method
# eval_manifest.json + derived_columns.csv, then run autolearn.feat_selection.eval_quality
# to score them against the same multi-model downstream sweep used for VI.
#
# Usage:
#   PYTHON=/path/to/python bash run_baselines.sh <dataset> <cuda> [total_iter] [top_k]
#
# Defaults: total_iter=10 (for INFER/LSBO), top_k=16 (passed to eval_quality)
#
# Output:
#   outputs/baselines/<dataset>/<method>/eval_manifest.json
#   outputs/baselines/<dataset>/<method>/derived_columns.csv
#   outputs/baselines/<dataset>/baselines_eval.csv  (multi-model downstream tables)
#
# Note: INFER and LSBO are run under score_mode=mi --lam=1.0 so they're
# directly comparable to VI@lam=1.0 (pure-usefulness search; only the
# inference strategy differs).

set -o pipefail

dataset=${1:?missing dataset}
cuda=${2:?missing cuda id}
total_iter=${3:-10}
top_k=${4:-16}

PYTHON=${PYTHON:-python}
BASELINES=("openfe" "autofeat" "eafe" "infer" "lsbo")
MODEL="RF"
N_JOBS=${N_JOBS:-8}
BASE_OUT="outputs/baselines/${dataset}"

mkdir -p "$BASE_OUT"

run_baseline() {
    local method=$1
    local out="$BASE_OUT/$method"
    mkdir -p "$out"
    echo ""
    echo "##### baseline=$method #####"
    local t0=$(date +%s)
    case "$method" in
        openfe)
            "$PYTHON" -m autolearn.feat_selection.openfe_baseline \
                --data "$dataset" --eval_model "$MODEL" --n_jobs "$N_JOBS" \
                --max_top_k "$top_k" --feat_pool "$out"
            ;;
        autofeat)
            "$PYTHON" -m autolearn.feat_selection.autofeat_baseline \
                --data "$dataset" --eval_model "$MODEL" --n_jobs "$N_JOBS" \
                --feat_pool "$out"
            ;;
        eafe)
            "$PYTHON" -m autolearn.feat_selection.eafe_baseline \
                --data "$dataset" --eval_model "$MODEL" \
                --use_cuda --feat_pool "$out"
            ;;
        infer|lsbo)
            # DIFER-family methods: run via iter_train.py under MI mode + lam=1.0
            # so the produced selected_topk_lam1.0.json drops in alongside the
            # other manifests.
            PYTHON="$PYTHON" bash run_iter.sh "$dataset" "$cuda" "$method" "$total_iter" "mi" "1.0"
            local src="outputs/nfo/${MODEL}/${method}/mi/lam1.0/${dataset}/selected_topk_lam1.0.json"
            if [ -f "$src" ]; then
                cp "$src" "$out/eval_manifest.json"
                echo "[$method] copied $src -> $out/eval_manifest.json"
            else
                echo "[$method] WARNING: expected $src not found"
                return 1
            fi
            ;;
        *)
            echo "[run_baselines] unknown method: $method"
            return 1
            ;;
    esac
    local elapsed=$(( $(date +%s) - t0 ))
    echo "[$method] done in ${elapsed}s"
}

# Run each baseline, continuing on failures so a missing dependency (e.g. AutoFeat
# not installed) does not block the rest.
for method in "${BASELINES[@]}"; do
    if ! run_baseline "$method"; then
        echo "[run_baselines] $method FAILED — continuing"
    fi
done

# Aggregate evaluation across all baselines that produced a manifest.
echo ""
echo "##### eval_quality across baselines #####"
"$PYTHON" -m autolearn.feat_selection.eval_quality \
    --data "$dataset" \
    --inputs "$BASE_OUT/*/eval_manifest.json" \
    --output "$BASE_OUT/baselines_eval.csv" \
    --top_k "$top_k"

echo ""
echo "==============================================================="
echo " Baseline outputs in: $BASE_OUT"
echo " Aggregate table:     $BASE_OUT/baselines_eval.csv"
echo ""
echo " To compare with your VI sweep results, run:"
echo "   $PYTHON -m autolearn.feat_selection.eval_quality \\"
echo "       --data $dataset \\"
echo "       --inputs '$BASE_OUT/*/eval_manifest.json' \\"
echo "                'outputs/quality_report/${dataset}_*/selected_topk_lam*.json' \\"
echo "       --output outputs/quality_report/${dataset}_compare.csv \\"
echo "       --top_k $top_k"
echo "==============================================================="
