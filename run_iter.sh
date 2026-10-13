#!/bin/bash

export PYTHONPATH=$PYTHONPATH:./
dataset=$1
cuda=$2
method=${3:-lsbo}      # lsbo | infer | vi
total_iter=${4:-10}    # number of NFO iterations
score_mode=${5:-mi}    # cv (legacy DIFER) | mi | pearson | fisher  (bi-objective B-MAFE for the last 3)
lam=${6:-1.0}          # only used when score_mode in {mi, pearson, fisher}: λ in S = λ·U + (1-λ)·E
model="RF"
PYTHON=${PYTHON:-python}  # override with PYTHON=/path/to/python bash run_iter.sh ...
HYPER_CONFIG=${HYPER_CONFIG:-default}  # set HYPER_CONFIG=difer_orig to use the LSTM-encoder original-DIFER preset

# Source-tree root — where iter_train.py lives. Keep this as the repo root.
SRC_ROOT="."

# Output-tree root — outputs_num/ vs outputs_numcat/ depending on the
# operator set the run uses. Detected from NFO_EXTRA_OPTS containing
# --num_ops_only, with an explicit OUTPUTS_ROOT override winning if set.
# Legacy callers that pass neither land on outputs_numcat/ (the historical
# default — full op set when the dataset has cat features).
if [ -n "${OUTPUTS_ROOT:-}" ]; then
    OUT_ROOT="${OUTPUTS_ROOT}"
elif [[ "${NFO_EXTRA_OPTS:-}" == *--num_ops_only* ]]; then
    OUT_ROOT="./outputs_num"
else
    OUT_ROOT="./outputs_numcat"
fi
mkdir -p "$OUT_ROOT"
# Optional tag to namespace output dirs (e.g. NFO_RUN_TAG=gat to keep GAT-
# encoder runs from overwriting cached LSTM-encoder baselines).
#   cv-mode: outputs/.../iter/      ->  outputs/.../iter_<tag>/
#   mi-mode: outputs/.../mi/lamX/   ->  outputs/.../mi_<tag>/lamX/
TAG=${NFO_RUN_TAG:-}
ITER_DIR=${TAG:+iter_$TAG}
ITER_DIR=${ITER_DIR:-iter}
MI_DIR=${TAG:+mi_$TAG}
MI_DIR=${MI_DIR:-mi}
CKPS_DIR=${TAG:+ckps_$TAG}
CKPS_DIR=${CKPS_DIR:-ckps}

# Both mi and pearson are bi-objective (lam-dependent) modes — separate
# output dirs per λ so caches don't collide. The MI_DIR prefix ("mi_<tag>")
# is reused for both; the tag is what distinguishes the usefulness metric
# (e.g. gat = MI, gat_pearson_mdl = Pearson + MDL).
case "$score_mode" in mi|pearson|fisher|downstream|split|split_ca|anom_auc|anom_tail|subgroup|subgroup_op|rule_out|g3out|normout|normg3|ifout) composite_mode=1;; *) composite_mode=0;; esac
if [ "$composite_mode" = "1" ]; then
    POOL_DIR="$OUT_ROOT/nfo/$model/$method/$MI_DIR/lam${lam}/$dataset"
    CKP_DIR="$OUT_ROOT/nfo/$model/$method/$MI_DIR/lam${lam}/$CKPS_DIR/$dataset"
else
    POOL_DIR="$OUT_ROOT/nfo/$model/$method/$ITER_DIR/$dataset"
    CKP_DIR="$OUT_ROOT/nfo/$model/$method/$CKPS_DIR/$dataset"
fi

OPTS=""
OPTS+=" --eval_model=$model"
OPTS+=" --data=$dataset"
OPTS+=" --cuda=$cuda"
OPTS+=" --hyper_config=$HYPER_CONFIG"
OPTS+=" --method=$method"
OPTS+=" --total_iter=$total_iter"
OPTS+=" --score_mode=$score_mode"
OPTS+=" --lam=$lam"
OPTS+=" --feat_pool=$POOL_DIR"
OPTS+=" --ckp_path=$CKP_DIR"

OPTS+=" --parallel_eval"

# NFO_EXTRA_OPTS lets the caller append arbitrary iter_train.py flags
# (e.g. NFO_EXTRA_OPTS="--num_ops_only") without changing this script.
# Quoted carefully so multi-word values still split into separate args.
$PYTHON ${SRC_ROOT}/autolearn/feat_selection/nfo/iter_train.py ${OPTS} ${NFO_EXTRA_OPTS:-}
