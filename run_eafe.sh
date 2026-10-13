#!/bin/bash
# Run the E-AFE baseline (PyTorch reimplementation of "Toward Efficient
# Automated Feature Engineering", Wang et al. ICDE 2023).
#
# Usage:
#   bash run_eafe.sh <dataset> [stage2_epochs]
#
# Output: outputs/nfo/RF/eafe/iter/<dataset>/results.csv (iter,score,top_k,method)
#
# Before running, train the FPE classifier once:
#   python -m autolearn.feat_selection.eafe.fpe_train --mode bootstrap \
#       --output models/fpe_classifier.pt --signature_dim 48
# If models/fpe_classifier.pt is missing, the run falls back to Stage 2 only
# (equivalent to NFS without pre-evaluation).

export PYTHONPATH=$PYTHONPATH:./
# OUTPUTS_ROOT: set by run_quality_compare.sh to route into outputs_num/ or
# outputs_numcat/; default outputs_numcat/ for standalone invocations.
OUTPUTS_ROOT=${OUTPUTS_ROOT:-./outputs_numcat}
BASE_PATH="."
dataset=$1
stage2_epochs=${2:-100}
model="RF"

OPTS=""
OPTS+=" --data=$dataset"
OPTS+=" --eval_model=$model"
OPTS+=" --stage2_epochs=$stage2_epochs"
OPTS+=" --fpe_model=$BASE_PATH/models/fpe_classifier.pt"
OPTS+=" --feat_pool=$OUTPUTS_ROOT/nfo/$model/eafe/iter/$dataset"

${PYTHON:-python} ${BASE_PATH}/autolearn/feat_selection/eafe_baseline.py ${OPTS}
