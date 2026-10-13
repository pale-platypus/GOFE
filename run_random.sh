#!/bin/bash
# Uniform random-expression pool generator, used to pad short methods to a
# common top-k (not a compared method). Samples N random feature trees and
# emits the same eval_manifest.json shape as the other method runners, so it
# slots directly into autolearn.feat_selection.eval_quality.
#
# Usage: bash run_random.sh <dataset> [n_features] [seed]

export PYTHONPATH=$PYTHONPATH:./
# OUTPUTS_ROOT: set by run_quality_compare.sh to route into outputs_num/ or
# outputs_numcat/; default outputs_numcat/ for standalone invocations.
OUTPUTS_ROOT=${OUTPUTS_ROOT:-./outputs_numcat}
BASE_PATH="."
dataset=$1
n_features=${2:-16}
seed=${3:-0}
model="RF"

OPTS=""
OPTS+=" --data=$dataset"
OPTS+=" --n_features=$n_features"
OPTS+=" --seed=$seed"
OPTS+=" --feat_pool=$OUTPUTS_ROOT/nfo/$model/random/iter/$dataset"

${PYTHON:-python} ${BASE_PATH}/autolearn/feat_selection/random_baseline.py ${OPTS}
