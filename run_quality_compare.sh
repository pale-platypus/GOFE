#!/bin/bash
# Usage:
#   PYTHON=/path/to/python bash run_quality_compare.sh <cuda> [total_iter] [top_k]
# Defaults: total_iter=10, top_k=16

set -o pipefail

cuda=${1:?missing cuda id}
total_iter=${2:-10}
top_k=${3:-16}

PYTHON=${PYTHON:-python}
MODEL="RF"
FORCE_RERUN=${FORCE_RERUN:-0}

DATASETS=(
    pima # yes
    german   # yes
    winequality-red  # yes
    # winequality-white # ?
    spambase  # yes
    amazon # yes
    housing  # yes
    airfoil # yes
    # 586   # no
    # 607  # no
# Override DATASETS from the environment when DATASETS_OVERRIDE is set.
# Lets you fill in only the missing cells without editing this file:
#   DATASETS_OVERRIDE="amazon housing airfoil credit_default 620 637" \
#     bash run_quality_compare.sh 0 10 16
# The existing skip-if-cached check still applies, so cells already on disk
# (e.g. autofeat manifest for housing) are skipped automatically — only
# genuinely missing cells (e.g. openfe@housing, lsbo_pearson_mdl@<6 dsets>)
# actually run.
if [ -n "${DATASETS_OVERRIDE:-}" ]; then
    # shellcheck disable=SC2206
    DATASETS=(${DATASETS_OVERRIDE})
    echo "[sweep] DATASETS_OVERRIDE set; datasets = ${DATASETS[*]}" >&2
fi
LAMS=(0.0 0.5 1.0)

# NFO_LAM_METHODS=(lsbo vi)              # depend on lam
# Each entry is a logical experiment ID, parsed by nfo_lam_spec() below
# into (real_method, score_mode, NFO_RUN_TAG, NFO_EXTRA_OPTS, HYPER_CONFIG).
# Adding a new metric pair = add an entry here + a case in the parser.
NFO_LAM_METHODS=(
    lsbo
    lsbo_pearson_mdl
    lsbo_fisher_mdl
    lsbo_fisher
)
# DIFER cells are lam-INDEPENDENT (score_mode=cv → no λ in path).
# They run in the per-dataset prep block, not the per-λ loop.
#   `infer_difer_orig` (numcat, full op set) — existing legacy DIFER baseline
#   `infer_cv`         (num,    --num_ops_only) — DIFER restricted to num ops
BASELINE_METHODS=(
    autofeat
    openfe
    eafe
    random
)
RANDOM_SEED=${RANDOM_SEED:-0}

NUM_ROOT="outputs_num"
NUMCAT_ROOT="outputs_numcat"
mkdir -p "$NUM_ROOT/quality_compare" "$NUMCAT_ROOT/quality_compare"

# NUM_ONLY=1 restricts this whole sweep invocation to outputs_num/:
#   - NFO_LAM_METHODS is filtered to cells whose spec contains --num_ops_only
#     (skipping plain `lsbo` MI cell which targets numcat)
#   - The legacy DIFER infer (cv-mode, full-op-set) cell is skipped
#   - Baselines write primarily into outputs_num/ and skip the numcat mirror
#   - Only the op_set='num' eval cell runs (no numcat eval)
#   - sweep.log + dataset.log live under outputs_num/quality_compare/
# Usage: NUM_ONLY=1 DATASETS_OVERRIDE="amazon housing ..." bash run_quality_compare.sh 0 10 16
NUM_ONLY=${NUM_ONLY:-0}
if [ "$NUM_ONLY" = "1" ]; then
    SWEEP_LOG="$NUM_ROOT/quality_compare/sweep.log"
else
    # Canonical home for the cross-tree top-level logs is the numcat tree.
    SWEEP_LOG="$NUMCAT_ROOT/quality_compare/sweep.log"
fi

# Resolve which tree a given experiment ID writes into.
nfo_outputs_root() {
    nfo_lam_spec "$1" || return 1
    if [[ "$NFO_EXTRA_OPTS" == *--num_ops_only* ]]; then
        echo "$NUM_ROOT"
    else
        echo "$NUMCAT_ROOT"
    fi
}

# Manifest paths used both for skip-if-exists and as eval_quality inputs.
# $1 = dataset, $2 = root (defaults to numcat — the legacy DIFER infer cell
# uses the full op set so its manifest lives there).
infer_manifest_path() {
    local root="${2:-$NUMCAT_ROOT}"
    echo "${root}/nfo/${MODEL}/infer/iter_difer_orig/$1/eval_manifest.json"
}
# $1 = method, $2 = dataset, $3 = root (defaults to numcat — baselines have
# no num/cat affinity; the mirror helper below pushes a copy into the other
# tree so both eval cells find them).
baseline_manifest_path() {
    local root="${3:-$NUMCAT_ROOT}"
    echo "${root}/nfo/${MODEL}/$1/iter/$2/eval_manifest.json"
}

# After a baseline finishes writing into one tree, copy the directory into
# the other so the parallel eval cell finds the same manifest without
# rerunning the baseline. Idempotent — falls through if src is missing.
_mirror_baseline_to_other_root() {
    # $1 = method, $2 = dataset, $3 = source root that just got written.
    local method=$1 dataset=$2 src_root=$3
    local other_root
    if [ "$src_root" = "$NUM_ROOT" ]; then
        other_root="$NUMCAT_ROOT"
    else
        other_root="$NUM_ROOT"
    fi
    local src_dir="${src_root}/nfo/${MODEL}/${method}/iter/${dataset}"
    local dst_dir="${other_root}/nfo/${MODEL}/${method}/iter/${dataset}"
    if [ -d "$src_dir" ]; then
        mkdir -p "$(dirname "$dst_dir")"
        # cp -a preserves timestamps so the eval cell's skip-if-newer logic
        # doesn't accidentally treat the mirror as fresh work to redo.
        cp -a "$src_dir" "$(dirname "$dst_dir")/"
    fi
}
# Map a logical experiment ID (NFO_LAM_METHODS entry) to its dispatch tuple.
# Writes its result into global vars NFO_REAL_METHOD / NFO_SCORE_MODE /
# NFO_TAG / NFO_EXTRA_OPTS / NFO_HYPER so the caller can read them after.
# The output dir under run_iter.sh becomes
#   mi-mode:  outputs/nfo/${MODEL}/<real_method>/mi_<tag>/lam<lam>/<dataset>/
#   cv-mode:  outputs/nfo/${MODEL}/<real_method>/iter_<tag>/<dataset>/
# so the tag is the only thing distinguishing experiment cells per method.
nfo_lam_spec() {
    case "$1" in
        lsbo)
            NFO_REAL_METHOD=lsbo
            NFO_SCORE_MODE=mi
            NFO_TAG=gat
            NFO_EXTRA_OPTS=""
            NFO_HYPER=default
            ;;
        lsbo_pearson_mdl)
            NFO_REAL_METHOD=lsbo
            NFO_SCORE_MODE=pearson
            NFO_TAG=gat_pearson_mdl
            NFO_EXTRA_OPTS="--num_ops_only --method_label=lsbo_pearson_mdl"
            NFO_HYPER=default
            ;;
        lsbo_fisher_mdl)
            # B-MAFE w/ Fisher score as usefulness, Kext as complexity,
            # numerical-only operators. Output:
            #   outputs_num/nfo/RF/lsbo/mi_gat_fisher_mdl/lam<λ>/<dataset>/
            NFO_REAL_METHOD=lsbo
            NFO_SCORE_MODE=fisher
            NFO_TAG=gat_fisher_mdl
            NFO_EXTRA_OPTS="--num_ops_only --method_label=lsbo_fisher_mdl"
            NFO_HYPER=default
            ;;
        lsbo_fisher)
            # B-MAFE w/ Fisher + Kext on the FULL operator set (num + cat ops).
            # Lands in the numcat tree because there is no --num_ops_only here.
            # Output:
            #   outputs_numcat/nfo/RF/lsbo/mi_gat_fisher/lam<λ>/<dataset>/
            NFO_REAL_METHOD=lsbo
            NFO_SCORE_MODE=fisher
            NFO_TAG=gat_fisher
            NFO_EXTRA_OPTS="--method_label=lsbo_fisher"
            NFO_HYPER=default
            ;;
        # `infer_pearson_mdl` removed: it ran INFER over a Pearson+Kext composite
        # surrogate, which is *not* what DIFER does. True DIFER's surrogate
        # predicts downstream CV; the `infer_cv` lam-independent cell below is
        # the honest DIFER baseline for the num tree (score_mode=cv, LSTM,
        # --num_ops_only). See the per-dataset prep block.
        *)
            echo "[sweep] ERROR: unknown NFO_LAM_METHODS entry '$1'" >&2
            return 1
            ;;
    esac
}

nfo_lam_manifest_path() {
    # $1 = logical experiment ID (NFO_LAM_METHODS entry), $2 = dataset, $3 = lam.
    # Path mirrors run_iter.sh's POOL_DIR + iter_train.py's selected_topk filename.
    # The root (outputs_num vs outputs_numcat) comes from the spec's --num_ops_only.
    # We call nfo_lam_spec() directly so NFO_* globals are populated even when
    # the caller did not — calling nfo_outputs_root in a $() subshell would
    # set them inside the subshell only and leave the parent stale.
    nfo_lam_spec "$1" || return 1
    local root
    if [[ "$NFO_EXTRA_OPTS" == *--num_ops_only* ]]; then
        root="$NUM_ROOT"
    else
        root="$NUMCAT_ROOT"
    fi
    if [ "$NFO_SCORE_MODE" = "mi" ] || [ "$NFO_SCORE_MODE" = "pearson" ] || [ "$NFO_SCORE_MODE" = "fisher" ]; then
        echo "${root}/nfo/${MODEL}/${NFO_REAL_METHOD}/mi_${NFO_TAG}/lam$3/$2/selected_topk_lam$3.json"
    else
        echo "${root}/nfo/${MODEL}/${NFO_REAL_METHOD}/iter_${NFO_TAG}/$2/eval_manifest.json"
    fi
}

# Overwrite the top-level log on each invocation (it is not timestamped).
: > "$SWEEP_LOG"
{
    echo "================================================================="
    echo "Quality comparison sweep (NFO-GAT vs external baselines)"
    echo "  datasets    = ${DATASETS[*]}"
    echo "  lambdas     = ${LAMS[*]}   (lsbo, vi)"
    echo "                infer (LSTM, cv, full-op-set) is the numcat baseline; lam-independent"
    echo "  NFO cells   = ${NFO_LAM_METHODS[*]}"
    echo "  cuda        = $cuda"
    echo "  total_iter  = $total_iter"
    echo "  top_k eval  = $top_k"
    echo "  python      = $PYTHON"
    echo "  num tree    = $NUM_ROOT"
    echo "  numcat tree = $NUMCAT_ROOT"
    echo "  NUM_ONLY    = $NUM_ONLY  (1 = skip everything that targets outputs_numcat)"
    echo "  sweep log   = $SWEEP_LOG"
    echo "  FORCE_RERUN = $FORCE_RERUN"
    echo "================================================================="
} | tee -a "$SWEEP_LOG"

sweep_t0=$(date +%s)

for dataset in "${DATASETS[@]}"; do
    echo "" | tee -a "$SWEEP_LOG"
    echo "#################################################################" | tee -a "$SWEEP_LOG"
    echo "##### DATASET: $dataset" | tee -a "$SWEEP_LOG"
    echo "#################################################################" | tee -a "$SWEEP_LOG"

    # Per-dataset log: numcat tree is the canonical home, but under NUM_ONLY=1
    # we keep everything under outputs_num/ so a re-run touches nothing else.
    if [ "$NUM_ONLY" = "1" ]; then
        DSET_LOG="$NUM_ROOT/quality_compare/$dataset.dataset.log"
    else
        DSET_LOG="$NUMCAT_ROOT/quality_compare/$dataset.dataset.log"
    fi
    mkdir -p "$(dirname "$DSET_LOG")"
    : > "$DSET_LOG"

    # ---- Per-dataset (lambda-independent) work -----------------------------
    echo "" | tee -a "$DSET_LOG"
    echo "##### Per-dataset prep: infer (original DIFER) + external baselines #####" | tee -a "$DSET_LOG"

    # INFER = original DIFER: LSTM encoder + score_mode=cv (downstream CV
    # objective) + full op set. Lives in the numcat tree.
    # Skipped entirely under NUM_ONLY=1 — it's a numcat-only cell.
    infer_src=""
    if [ "$NUM_ONLY" != "1" ]; then
        infer_src=$(infer_manifest_path "$dataset" "$NUMCAT_ROOT")
        echo "" | tee -a "$DSET_LOG"
        echo "----- infer (original DIFER: LSTM encoder, score_mode=cv) -> $NUMCAT_ROOT -----" | tee -a "$DSET_LOG"
        if [ -f "$infer_src" ] && [ "$FORCE_RERUN" != "1" ]; then
            echo "[run] infer SKIPPED (cached manifest at $infer_src)" | tee -a "$DSET_LOG"
        else
            t0=$(date +%s)
            if PYTHON="$PYTHON" NFO_RUN_TAG=difer_orig HYPER_CONFIG=difer_orig \
                    NFO_EXTRA_OPTS="" bash run_iter.sh \
                    "$dataset" "$cuda" "infer" "$total_iter" "cv" "1.0" \
                    2>&1 | tee -a "$DSET_LOG"; then
                :
            else
                echo "[run] infer FAILED (continuing)" | tee -a "$DSET_LOG"
            fi
            elapsed=$(( $(date +%s) - t0 ))
            echo "[run] infer done in ${elapsed}s" | tee -a "$DSET_LOG"
        fi
        if [ ! -f "$infer_src" ]; then
            echo "[run] WARNING: missing $infer_src" | tee -a "$DSET_LOG"
            infer_src=""
        fi
    else
        echo "" | tee -a "$DSET_LOG"
        echo "----- infer (legacy DIFER, numcat) SKIPPED — NUM_ONLY=1 -----" | tee -a "$DSET_LOG"
    fi

    # DIFER on num ops only (cv-mode, LSTM encoder) — the honest DIFER row for
    # the num tree. Lam-INDEPENDENT (score_mode=cv → no λ in path). Replaces
    # the now-removed `infer_pearson_mdl` cell which mis-attributed B-MAFE's
    # composite surrogate to DIFER.
    infer_cv_src="$NUM_ROOT/nfo/${MODEL}/infer/iter_difer_orig/$dataset/eval_manifest.json"
    echo "" | tee -a "$DSET_LOG"
    echo "----- infer_cv (DIFER: LSTM encoder, score_mode=cv, --num_ops_only) -> $NUM_ROOT -----" | tee -a "$DSET_LOG"
    if [ -f "$infer_cv_src" ] && [ "$FORCE_RERUN" != "1" ]; then
        echo "[run] infer_cv SKIPPED (cached manifest at $infer_cv_src)" | tee -a "$DSET_LOG"
    else
        t0=$(date +%s)
        if PYTHON="$PYTHON" NFO_RUN_TAG=difer_orig HYPER_CONFIG=difer_orig \
                NFO_EXTRA_OPTS="--num_ops_only --method_label=infer_cv" bash run_iter.sh \
                "$dataset" "$cuda" "infer" "$total_iter" "cv" "1.0" \
                2>&1 | tee -a "$DSET_LOG"; then
            :
        else
            echo "[run] infer_cv FAILED (continuing)" | tee -a "$DSET_LOG"
        fi
        elapsed=$(( $(date +%s) - t0 ))
        echo "[run] infer_cv done in ${elapsed}s" | tee -a "$DSET_LOG"
    fi
    if [ ! -f "$infer_cv_src" ]; then
        echo "[run] WARNING: missing $infer_cv_src" | tee -a "$DSET_LOG"
        infer_cv_src=""
    fi

    # Baselines are op-set-agnostic. Under NUM_ONLY=1 we write primary into
    # outputs_num/ and skip the numcat mirror; otherwise outputs_numcat/ is
    # primary and we mirror to outputs_num/.
    declare -A baseline_manifests_numcat=() baseline_manifests_num=()
    if [ "$NUM_ONLY" = "1" ]; then
        bl_primary_root="$NUM_ROOT"
    else
        bl_primary_root="$NUMCAT_ROOT"
    fi
    for m in "${BASELINE_METHODS[@]}"; do
        echo "" | tee -a "$DSET_LOG"
        if [ "$NUM_ONLY" = "1" ]; then
            echo "----- baseline $m (primary -> $NUM_ROOT, no mirror — NUM_ONLY=1) -----" | tee -a "$DSET_LOG"
        else
            echo "----- baseline $m (primary -> $NUMCAT_ROOT, mirror -> $NUM_ROOT) -----" | tee -a "$DSET_LOG"
        fi
        manifest_primary=$(baseline_manifest_path "$m" "$dataset" "$bl_primary_root")
        manifest_other=""
        if [ "$NUM_ONLY" != "1" ]; then
            manifest_other=$(baseline_manifest_path "$m" "$dataset" "$NUM_ROOT")
        fi
        # Skip if primary exists and (NUM_ONLY=1 OR mirror exists too).
        if [ -f "$manifest_primary" ] \
                && { [ "$NUM_ONLY" = "1" ] || [ -f "$manifest_other" ]; } \
                && [ "$FORCE_RERUN" != "1" ]; then
            echo "[baseline] $m SKIPPED (cached at $manifest_primary)" | tee -a "$DSET_LOG"
        else
            t0=$(date +%s)
            case "$m" in
                autofeat) PYTHON="$PYTHON" OUTPUTS_ROOT="$bl_primary_root" bash run_autofeat.sh "$dataset" 2>&1 | tee -a "$DSET_LOG" ;;
                openfe)   PYTHON="$PYTHON" OUTPUTS_ROOT="$bl_primary_root" bash run_openfe.sh   "$dataset" 2>&1 | tee -a "$DSET_LOG" ;;
                eafe)     PYTHON="$PYTHON" OUTPUTS_ROOT="$bl_primary_root" bash run_eafe.sh     "$dataset" 2>&1 | tee -a "$DSET_LOG" ;;
                random)   PYTHON="$PYTHON" OUTPUTS_ROOT="$bl_primary_root" bash run_random.sh   "$dataset" "$top_k" "$RANDOM_SEED" 2>&1 | tee -a "$DSET_LOG" ;;
            esac
            elapsed=$(( $(date +%s) - t0 ))
            echo "[baseline] $m done in ${elapsed}s" | tee -a "$DSET_LOG"
            if [ "$NUM_ONLY" != "1" ]; then
                _mirror_baseline_to_other_root "$m" "$dataset" "$NUMCAT_ROOT"
                echo "[baseline] $m mirrored to $NUM_ROOT" | tee -a "$DSET_LOG"
            fi
        fi
        # Fill the relevant map(s) so the per-op_set eval loop can find them.
        if [ "$NUM_ONLY" = "1" ]; then
            [ -f "$manifest_primary" ] && baseline_manifests_num[$m]="$manifest_primary"
            if [ ! -f "$manifest_primary" ]; then
                echo "[baseline] WARNING: missing $manifest_primary" | tee -a "$DSET_LOG"
            fi
        else
            [ -f "$manifest_primary" ] && baseline_manifests_numcat[$m]="$manifest_primary"
            [ -f "$manifest_other" ]   && baseline_manifests_num[$m]="$manifest_other"
            if [ ! -f "$manifest_primary" ]; then
                echo "[baseline] WARNING: missing $manifest_primary" | tee -a "$DSET_LOG"
            fi
            if [ ! -f "$manifest_other" ]; then
                echo "[baseline] WARNING: missing $manifest_other (mirror may have failed)" | tee -a "$DSET_LOG"
            fi
        fi
    done

    # ---- Per-lambda work ---------------------------------------------------
    # When no λ-dependent methods are configured (VI/LSBO both off), the inputs
    # to eval_quality are the same for every λ — looping would just write
    # identical tables.csv into lam<λ>/ three times. Collapse to a single cell
    # at LAMS[0] in that case. The eval still passes the full LAMS grid via
    # --lambdas, so composite_mean_lam<λ> columns remain populated.
    if [ ${#NFO_LAM_METHODS[@]} -eq 0 ]; then
        LAMS_ACTIVE=("${LAMS[0]}")
        echo "[run] no λ-dependent methods configured; collapsing per-λ loop to lam=${LAMS_ACTIVE[0]} only" \
            | tee -a "$DSET_LOG"
    else
        LAMS_ACTIVE=("${LAMS[@]}")
    fi

    for lam in "${LAMS_ACTIVE[@]}"; do
        echo "" | tee -a "$DSET_LOG"
        echo "=================================================================" | tee -a "$DSET_LOG"
        echo "===== $dataset / lam = $lam =====" | tee -a "$DSET_LOG"
        echo "=================================================================" | tee -a "$DSET_LOG"

        # Partition NFO cells by their target op-set tree. Each partition
        # gets its own eval cell so tables.csv lives next to the manifests
        # it actually consumes.
        cells_numcat=()
        cells_num=()
        for m in "${NFO_LAM_METHODS[@]}"; do
            if ! nfo_lam_spec "$m"; then
                continue
            fi
            if [[ "$NFO_EXTRA_OPTS" == *--num_ops_only* ]]; then
                cells_num+=("$m")
            else
                cells_numcat+=("$m")
            fi
        done

        # Under NUM_ONLY=1 we only run the num eval cell — skip numcat entirely.
        if [ "$NUM_ONLY" = "1" ]; then
            OP_SETS=("num")
        else
            OP_SETS=("numcat" "num")
        fi
        for op_set in "${OP_SETS[@]}"; do
            if [ "$op_set" = "numcat" ]; then
                op_root="$NUMCAT_ROOT"
                op_cells=("${cells_numcat[@]}")
            else
                op_root="$NUM_ROOT"
                op_cells=("${cells_num[@]}")
            fi
            # Skip op-sets that have no per-λ NFO cells AND no relevant DIFER
            # cell. (The DIFER cells are lam-independent; checked separately.)
            if [ ${#op_cells[@]} -eq 0 ]; then
                difer_present=0
                if [ "$op_set" = "numcat" ] && [ -n "$infer_src" ]; then difer_present=1; fi
                if [ "$op_set" = "num" ]    && [ -n "$infer_cv_src" ]; then difer_present=1; fi
                if [ "$difer_present" = "0" ]; then
                    echo "[eval] $op_set has no NFO cells and no DIFER manifest; skipping" | tee -a "$DSET_LOG"
                    continue
                fi
            fi

            LAM_DIR="${op_root}/quality_compare/$dataset/lam${lam}"
            mkdir -p "$LAM_DIR"
            RUN_LOG="$LAM_DIR/run.log"
            EVAL_LOG="$LAM_DIR/eval.log"
            TABLES_CSV="$LAM_DIR/tables.csv"
            : > "$RUN_LOG"

            echo "" | tee -a "$RUN_LOG"
            echo "----- op_set = $op_set  (root = $op_root) -----" | tee -a "$RUN_LOG"

            inputs=()
            # DIFER manifests are full-op-set OR num-only depending on tree:
            #   numcat eval gets `infer_src` (legacy DIFER, full op set)
            #   num    eval gets `infer_cv_src` (DIFER w/ --num_ops_only)
            # Both are score_mode=cv (downstream-CV surrogate) — the honest
            # DIFER row for each tree.
            if [ "$op_set" = "numcat" ] && [ -n "$infer_src" ]; then
                inputs+=("$infer_src")
            fi
            if [ "$op_set" = "num" ] && [ -n "$infer_cv_src" ]; then
                inputs+=("$infer_cv_src")
            fi

            for m in "${op_cells[@]}"; do
                echo "" | tee -a "$RUN_LOG"
                echo "----- $m (lam=$lam, op_set=$op_set) -----" | tee -a "$RUN_LOG"
                nfo_lam_spec "$m"  # repopulate NFO_* globals for this cell
                src=$(nfo_lam_manifest_path "$m" "$dataset" "$lam")
                if [ -f "$src" ] && [ "$FORCE_RERUN" != "1" ]; then
                    echo "[run] $m SKIPPED (cached manifest at $src)" | tee -a "$RUN_LOG"
                else
                    t0=$(date +%s)
                    if PYTHON="$PYTHON" \
                            NFO_RUN_TAG="$NFO_TAG" \
                            HYPER_CONFIG="$NFO_HYPER" \
                            NFO_EXTRA_OPTS="$NFO_EXTRA_OPTS" \
                            bash run_iter.sh \
                            "$dataset" "$cuda" "$NFO_REAL_METHOD" \
                            "$total_iter" "$NFO_SCORE_MODE" "$lam" \
                            2>&1 | tee -a "$RUN_LOG"; then
                        :
                    else
                        echo "[run] $m FAILED (continuing)" | tee -a "$RUN_LOG"
                        continue
                    fi
                    elapsed=$(( $(date +%s) - t0 ))
                    echo "[run] $m done in ${elapsed}s" | tee -a "$RUN_LOG"
                fi
                if [ -f "$src" ]; then
                    inputs+=("$src")
                else
                    echo "[run] WARNING: missing $src" | tee -a "$RUN_LOG"
                fi
            done

            # Pull baselines from the matching tree (mirror was done earlier).
            if [ "$op_set" = "numcat" ]; then
                declare -n bl_map=baseline_manifests_numcat
            else
                declare -n bl_map=baseline_manifests_num
            fi
            for m in "${BASELINE_METHODS[@]}"; do
                if [ -n "${bl_map[$m]:-}" ]; then
                    inputs+=("${bl_map[$m]}")
                fi
            done
            unset -n bl_map

            if [ ${#inputs[@]} -eq 0 ]; then
                echo "[eval] no inputs for ($dataset, lam=$lam, op_set=$op_set); skipping" | tee -a "$RUN_LOG"
                continue
            fi

            echo "" | tee -a "$RUN_LOG"
            echo "##### eval_quality (3 tables) -> $TABLES_CSV #####" | tee -a "$RUN_LOG"
            echo "inputs:" | tee -a "$RUN_LOG"
            for inp in "${inputs[@]}"; do echo "  $inp" | tee -a "$RUN_LOG"; done

            PYTHONPATH=. "$PYTHON" -m autolearn.feat_selection.eval_quality \
                --data "$dataset" \
                --inputs "${inputs[@]}" \
                --output "$TABLES_CSV" \
                --checkpoint_dir "$LAM_DIR" \
                --include_base_features \
                --lambdas "${LAMS[@]}" \
                --top_k "$top_k" \
                ${EVAL_EXTRA_OPTS:-} \
                2>&1 | tee "$EVAL_LOG"
        done
    done

    # reset associative arrays for the next dataset
    unset baseline_manifests_numcat baseline_manifests_num
done

sweep_elapsed=$(( $(date +%s) - sweep_t0 ))
echo "" | tee -a "$SWEEP_LOG"
echo "===============================================================" | tee -a "$SWEEP_LOG"
echo " Sweep finished in ${sweep_elapsed}s"                            | tee -a "$SWEEP_LOG"
echo " Output layout:"                                                 | tee -a "$SWEEP_LOG"
echo "   ${NUMCAT_ROOT}/quality_compare/sweep.log               (cross-tree top-level)" | tee -a "$SWEEP_LOG"
echo "   ${NUMCAT_ROOT}/quality_compare/<dataset>.dataset.log   (per-dataset infer + baselines)" | tee -a "$SWEEP_LOG"
echo "   ${NUMCAT_ROOT}/quality_compare/<dataset>/lam<lam>/{run,eval}.log + tables.csv" | tee -a "$SWEEP_LOG"
echo "   ${NUM_ROOT}/quality_compare/<dataset>/lam<lam>/{run,eval}.log + tables.csv" | tee -a "$SWEEP_LOG"
echo "   ${NUMCAT_ROOT}/nfo/RF/<method>/...   (manifests for full-op-set runs + mirrored baselines)" | tee -a "$SWEEP_LOG"
echo "   ${NUM_ROOT}/nfo/RF/<method>/...      (manifests for num-only runs + mirrored baselines)" | tee -a "$SWEEP_LOG"
echo " Cached cells were SKIPPED; pass FORCE_RERUN=1 to override."     | tee -a "$SWEEP_LOG"
echo "===============================================================" | tee -a "$SWEEP_LOG"
