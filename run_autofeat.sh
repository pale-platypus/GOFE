#!/bin/bash
# Run the AutoFeat baseline on a dataset.
# Usage: bash run_autofeat.sh <dataset> [n_jobs]
#
# Output lands in outputs/nfo/RF/autofeat/iter/<dataset>/results.csv
# in the same iter,score,top_k,method shape as run_iter.sh's output,
# so run_compare.sh-style scripts can ingest it.

export PYTHONPATH=$PYTHONPATH:./
# Honor OUTPUTS_ROOT so the sweep can route baselines into outputs_num/ or
# outputs_numcat/. Baselines have no num/cat affinity (they run a fixed
# algorithm), so the sweep typically runs once and mirrors the resulting
# manifest into both trees — see _mirror_baseline_to_other_root() in
# run_quality_compare.sh.
OUTPUTS_ROOT=${OUTPUTS_ROOT:-./outputs_numcat}
BASE_PATH="."
dataset=$1
n_jobs=${2:-1}
model="RF"

# AutoFeat knobs.
#   feateng_steps=3  (depth of feature composition; upstream default is 2,
#                     we use 3 for stronger composition by default)
#   featsel_runs=3   (number of bootstrap selection passes; upstream default is 5,
#                     we use 3 to cut runtime)
# Override per-run via env vars, e.g. AF_FEATENG_STEPS=1 bash run_autofeat.sh ...
AF_FEATENG_STEPS=${AF_FEATENG_STEPS:-3}
AF_FEATSEL_RUNS=${AF_FEATSEL_RUNS:-3}

# Operator-set restriction. Defaults to the closest match to NFO's 4+5 set:
#   log, sqrt, 1/  (≈ NFO log, sqrt_abs, reciprocal) + implicit +,-,*,/
# AutoFeat has no analog for NFO's min_max or mod_column. Set
# AF_TRANSFORMATIONS="" to use AutoFeat's default (~13 unary ops).
AF_TRANSFORMATIONS=${AF_TRANSFORMATIONS-"log sqrt 1/"}

OPTS=""
OPTS+=" --data=$dataset"
OPTS+=" --eval_model=$model"
OPTS+=" --n_jobs=$n_jobs"
OPTS+=" --feateng_steps=$AF_FEATENG_STEPS"
OPTS+=" --featsel_runs=$AF_FEATSEL_RUNS"
OPTS+=" --feat_pool=$OUTPUTS_ROOT/nfo/$model/autofeat/iter/$dataset"
if [ -n "$AF_TRANSFORMATIONS" ]; then
    OPTS+=" --transformations=\"$AF_TRANSFORMATIONS\""
fi
if [ -n "${FAIR_TEST:-}" ]; then OPTS+=" --test_data=$FAIR_TEST"; fi

eval "${PYTHON:-python} ${BASE_PATH}/autolearn/feat_selection/autofeat_baseline.py ${OPTS}"
