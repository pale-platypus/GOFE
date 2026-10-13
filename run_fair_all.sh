#!/bin/bash
# FAIR comparison: regenerate every method's features on the TRAIN split only
# (<name>__tr), writing into outputs_fair/. The held-out test (<name>__te) is
# NEVER touched here. Config is frozen (no tuning on results). Resumable:
# each method is skipped if its manifest already exists.
#
# Usage: PYTHON=/path/to/python bash run_fair_all.sh
export PYTHONPATH="${PYTHONPATH:-}:./"
export PYTHONIOENCODING=utf-8
export OUTPUTS_ROOT="./outputs_fair"
PY=${PYTHON:-python}
TOTAL_ITER=10
# Dataset bases via args (default: original 4). Pass e.g. `bash run_fair_all.sh
# winequality-red spambase ...`. RUN_DSLIN_MI=0 skips the DS-LIN and MI variants
# (we standardized on DS-RF) to save compute on large datasets.
DATASETS=(pima german housing 620)
if [ $# -gt 0 ]; then DATASETS=("$@"); fi
RUN_DSLIN_MI=${RUN_DSLIN_MI:-1}

run_if_missing() {  # $1=manifest  $2=label  (rest = command via env+args)
  local manifest="$1"; local label="$2"; shift 2
  if [ -f "$manifest" ]; then
    echo "[skip] $label (cached: $manifest)"; return 0
  fi
  echo "===== RUN $label ====="
  local t0; t0=$(date +%s)
  "$@"
  echo "[done] $label in $(( $(date +%s)-t0 ))s -> $( [ -f "$manifest" ] && echo OK || echo MISSING )"
}

for base in "${DATASETS[@]}"; do
  D="${base}__tr"
  echo ""; echo "############### DATASET $D ###############"

  # --- B-MAFE(DS): downstream search optimizing RF (the standardized variant) ---
  run_if_missing "outputs_fair/nfo/RF/lsbo/mi_gat_downstream/lam1.0/$D/selected_topk_lam1.0.json" \
    "B-MAFE(DS-RF) $D" \
    env PYTHON="$PY" HYPER_CONFIG=default_downstream NFO_RUN_TAG=gat_downstream \
        NFO_EXTRA_OPTS="--method_label=lsbo_downstream --new_feat=64 --downstream_model=RF" \
        bash run_iter.sh "$D" 0 lsbo "$TOTAL_ITER" downstream 1.0

  if [ "$RUN_DSLIN_MI" = "1" ]; then
    # --- B-MAFE(DS-LIN): downstream search optimizing the linear model ---
    run_if_missing "outputs_fair/nfo/RF/lsbo/mi_gat_downstream_lin/lam1.0/$D/selected_topk_lam1.0.json" \
      "B-MAFE(DS-LIN) $D" \
      env PYTHON="$PY" HYPER_CONFIG=default_downstream NFO_RUN_TAG=gat_downstream_lin \
          NFO_EXTRA_OPTS="--method_label=lsbo_downstream_lin --new_feat=64 --downstream_model=LIN" \
          bash run_iter.sh "$D" 0 lsbo "$TOTAL_ITER" downstream 1.0
    # --- B-MAFE(MI): bi-objective MI proxy ---
    run_if_missing "outputs_fair/nfo/RF/lsbo/mi_gat/lam1.0/$D/selected_topk_lam1.0.json" \
      "B-MAFE(MI) $D" \
      env PYTHON="$PY" HYPER_CONFIG=default NFO_RUN_TAG=gat \
          NFO_EXTRA_OPTS="--num_ops_only --method_label=lsbo" \
          bash run_iter.sh "$D" 0 lsbo "$TOTAL_ITER" mi 1.0
  fi

  # --- DIFER (infer, cv-mode, LSTM encoder) ---
  run_if_missing "outputs_fair/nfo/RF/infer/iter_difer_orig/$D/eval_manifest.json" \
    "DIFER $D" \
    env PYTHON="$PY" HYPER_CONFIG=difer_orig NFO_RUN_TAG=difer_orig \
        NFO_EXTRA_OPTS="--num_ops_only --method_label=infer_cv" \
        bash run_iter.sh "$D" 0 infer "$TOTAL_ITER" cv 1.0

  # --- baselines (OpenFE/AutoFeat get FAIR_TEST so they emit held-out test cols;
  #     skip key is derived_test_columns.csv, not the train manifest) ---
  run_if_missing "outputs_fair/nfo/RF/openfe/iter/$D/derived_test_columns.csv" \
    "OpenFE $D"   env PYTHON="$PY" OUTPUTS_ROOT=./outputs_fair FAIR_TEST="${base}__te" bash run_openfe.sh   "$D"
  run_if_missing "outputs_fair/nfo/RF/autofeat/iter/$D/derived_test_columns.csv" \
    "AutoFeat $D" env PYTHON="$PY" OUTPUTS_ROOT=./outputs_fair FAIR_TEST="${base}__te" bash run_autofeat.sh "$D"
  run_if_missing "outputs_fair/nfo/RF/eafe/iter/$D/eval_manifest.json" \
    "E-AFE $D"    env PYTHON="$PY" OUTPUTS_ROOT=./outputs_fair bash run_eafe.sh     "$D"
  run_if_missing "outputs_fair/nfo/RF/random/iter/$D/eval_manifest.json" \
    "Random $D"   env PYTHON="$PY" OUTPUTS_ROOT=./outputs_fair bash run_random.sh   "$D" 16 0
done
echo ""; echo "############### ALL FAIR RUNS DONE ###############"
