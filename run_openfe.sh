#!/bin/bash
# Run the OpenFE baseline on a dataset.
# Usage: bash run_openfe.sh <dataset> [n_jobs]
#
# Output lands in outputs/nfo/RF/openfe/iter/<dataset>/results.csv
# in the same iter,score,top_k,method shape as run_iter.sh's output,
# so run_compare.sh-style scripts can ingest it.

export PYTHONPATH=$PYTHONPATH:./
# Honor OUTPUTS_ROOT (set by run_quality_compare.sh to outputs_num/ or
# outputs_numcat/); default is outputs_numcat/ for standalone invocations.
OUTPUTS_ROOT=${OUTPUTS_ROOT:-./outputs_numcat}
BASE_PATH="."
dataset=$1
# n_jobs default = 8 (Monday default). The ENOSPC seen earlier from
# concurrent LightGBM workers flooding the parent's stdout pipe is now
# blocked by an fd-level redirect inside ``openfe_baseline.run_openfe``
# (``os.dup2(devnull, 1)`` before ``ofe.fit``), so workers inherit NUL as
# fd 1 and can't saturate the parent's pipe. If ENOSPC returns, lower this.
n_jobs=${2:-8}
model="RF"

# Operator-set restriction. Defaults to the closest match to NFO's 4+5 set:
#   log, sqrt  (≈ NFO log, sqrt_abs)  +  +,-,*,/  (≈ NFO +,-,*,division)
# OpenFE has no analog for NFO's min_max or mod_column, and no unary reciprocal.
# Set OFE_RESTRICT_OPS="" to disable the restriction (use OpenFE defaults).
OFE_RESTRICT_OPS=${OFE_RESTRICT_OPS-"log,sqrt,+,-,*,/"}

OPTS=""
OPTS+=" --data=$dataset"
OPTS+=" --eval_model=$model"
OPTS+=" --n_jobs=$n_jobs"
OPTS+=" --feat_pool=$OUTPUTS_ROOT/nfo/$model/openfe/iter/$dataset"
if [ -n "$OFE_RESTRICT_OPS" ]; then
    OPTS+=" --restrict_ops=$OFE_RESTRICT_OPS"
fi
if [ -n "${FAIR_TEST:-}" ]; then OPTS+=" --test_data=$FAIR_TEST"; fi

${PYTHON:-python} ${BASE_PATH}/autolearn/feat_selection/openfe_baseline.py ${OPTS}
