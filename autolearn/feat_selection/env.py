import os
import copy
import contextlib
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed, parallel_backend


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager that routes joblib batch completions into a tqdm bar."""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()

# import weka.core.converters as converters
# import weka.core.jvm as jvm
# from weka.classifiers import Classifier, Evaluation
# from weka.core.classes import Random
# from weka.filters import Filter

from autolearn.data import Dataset
from autolearn.utils import timeit, log as logger
from autolearn.metrics import r2_score
from autolearn.feat_selection.nfo.search_space import *

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import make_scorer
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor


project_path = Path(os.path.abspath(os.path.dirname(__file__)))


def return_0():
    return 0


def return_false():
    return False


def return_num():
    return OpType.NUM


class Environment:
    def __init__(self, dataset_path, max_order=4, cv=5, parallel=True, use_cat_space=True,
                 force_num_ops=False, ops_subset=None):
        # use_cat_space defaults to True so every entry point picks up cat
        # operators (count_hash, nunique, cat2num_mean) on mixed datasets
        # without needing an explicit flag. The conditional below still gates
        # on dataset.meta, so pure-numeric datasets transparently fall back
        # to num_op_info — no behavior change for them.
        #
        # force_num_ops: when True, override both use_cat_space and dataset
        # meta and pin op_info to num_op_info. Cat features still exist in
        # the dataset; they just won't get cat-typed operators (count_hash,
        # nunique, cat2num_mean) generated on top. Used by experiments
        # (e.g., the Pearson sweep) that need an apples-to-apples numerical
        # operator space regardless of dataset composition.
        self.dataset_path = dataset_path
        self.dataset = Dataset(dataset_path)
        self.use_cat_space = use_cat_space and not force_num_ops
        self.force_num_ops = bool(force_num_ops)
        if force_num_ops:
            self.op_info = num_op_info
        elif 'cat' in set(self.dataset.meta.values()) and use_cat_space:
            self.op_info = all_op_info
        else:
            self.op_info = num_op_info
        # ops_subset: optional operator-name whitelist for the |O| vocabulary
        # ablation. Applied after the num/cat selection above so every derived
        # structure (op_map, arity, tokenizer vocab, Kext's log2|O| term)
        # shrinks consistently. Caches built with a different vocabulary are
        # not loadable — ablation runs must use fresh feat_pool/ckp dirs.
        if ops_subset:
            keep = list(dict.fromkeys(ops_subset))
            known = {op[0] for op in self.op_info}
            unknown = [o for o in keep if o not in known]
            if unknown:
                raise ValueError(
                    f"ops_subset contains unknown operators {unknown}; "
                    f"available: {sorted(known)}")
            self.op_info = [op for op in self.op_info if op[0] in set(keep)]
        self._op_map, self._ops = None, None
        self._op_arity, self._op_order = None, None
        self._type_dict = None
        self.max_order = max_order
        self.max_length = max(self.op_arity.values()) ** (self.max_order + 1) - 1
        self.cv = cv
        self.parallel = parallel

    @property
    def op_map(self):
        if self._op_map is None:
            self._op_map = {op[0]: op for op in self.op_info}
        return self._op_map
    
    @property
    def ops(self):
        if self._ops is None:
            self._ops = [op[0] for op in self.op_info]
        return self._ops

    @property
    def type_dict(self):
        if self._type_dict is None and self.use_cat_space:
            type_map = {'num': OpType.NUM, 'cat': OpType.CAT}
            self._type_dict = defaultdict(return_num)
            self._type_dict.update({
                key: type_map[value] for key, value in self.dataset.meta.items() if value in type_map
            })
            self._type_dict.update({
                op[0]: op[4] for op in self.op_info
            })
        elif self._type_dict is None:
            self._type_dict = defaultdict(return_num)
        return self._type_dict

    @property
    def op_arity(self):
        if self._op_arity is None:
            self._op_arity = defaultdict(return_0)
            for op in self.op_info:
                self._op_arity[op[0]] = op[1]
        return self._op_arity

    @property
    def op_order(self):
        if self._op_order is None:
            self._op_order = defaultdict(return_false)
            for op in self.op_info:
                self.op_order[op[0]] = op[2]
        return self._op_order

    @property
    def features(self):
        return list(map(str, self.dataset.features))

    def _weka_evaluate_r(self, data):
        data.class_is_last()
        model = Classifier(classname='weka.classifiers.trees.RandomForest')
        evl = Evaluation(data)
        evl.crossvalidate_model(model, data, self.cv, Random(0))
        s = 1 - evl.relative_absolute_error / 100
        return s

    def _weka_evaluate_c(self, data):
        weka_filter = Filter(
            classname="weka.filters.unsupervised.attribute.NumericToNominal",
            options=["-R", "last"]
        )
        weka_filter.inputformat(data)
        data = weka_filter.filter(data)
        data.class_is_last()
        model = Classifier(classname='weka.classifiers.trees.RandomForest')
        evl = Evaluation(data)
        evl.crossvalidate_model(model, data, self.cv, Random(0))
        fscore = evl.weighted_f_measure
        s = fscore
        return s

    def add_feature(self, x, feat, name='new'):
        # x.insert(0, name, feat)
        x[name] = feat
        return x

    def add_features(self, x, feats):
        for i, feat in enumerate(feats):
            self.add_feature(x, feat, name=f"new_{i}")
        return x

    def weka_evaluate(self, feats):
        if not jvm.started:
            jvm.start()
        x = copy.deepcopy(self.dataset.instances)
        y = copy.deepcopy(self.dataset.labels)
        x = self.add_features(x, feats)
        x['y'] = y
        d = x.shape[0]
        x = str(x.values.tolist())

        data = np.reshape(eval(x), [d, -1], order='C')
        data = data.astype(np.float64)
        data = converters.ndarray_to_instances(
            data, relation='tmp'
        )
        if self.dataset.task == 'regression':
            score = self._weka_evaluate_r(data)
        elif self.dataset.task == 'classification':
            score = self._weka_evaluate_c(data)
        else:
            score = -1
        return score

    def get_feat(self, feat):
        feat = int(feat)
        return np.asarray(self.dataset.instances[feat].values, dtype=np.float64)

    def get_op(self, op):
        return self.op_map[op][3]

    def construct_feature(self, feat):
        return feat.generate(self)

    def _evaluate(self, feats):
        return self.weka_evaluate(feats)

    def _seq_eval(self, features):
        scores = []
        for feature in tqdm(features):
            scores.append(self._evaluate([feature]))
        return scores

    def _parallel_eval(self, features):
        n_jobs = max(1, (os.cpu_count() or 2) - 1)
        with parallel_backend("multiprocessing", n_jobs=n_jobs):
            with tqdm_joblib(tqdm(total=len(features))):
                scores = Parallel()(
                    delayed(self._evaluate)([feature]) for feature in features
                )
        return scores

    # def fake_eval(self, features):
    #     return 0.0

    @timeit
    def eval_features(self, features):
        res = -np.ones(len(features), dtype=np.float64)
        useful_features = []
        translated_features = []
        for i, feature in enumerate(features):
            try:
                feat_i = self.construct_feature(feature)
                useful_features.append(i)
                translated_features.append(feat_i)
            except Exception as e:
                logger(f"Error in construct {feature} {e}", level='error')
        if self.parallel:
            scores = self._parallel_eval(translated_features)
        else:
            scores = self._seq_eval(translated_features)
        res[useful_features] = scores
        return res

    def eval_set(self, features):
        useful_feats = []
        for feat in features:
            try:
                feat = self.construct_feature(feat)
                useful_feats.append(feat)
            except Exception as e:
                logger(f"Error in construct {feat} {e}", level='error')
        score = self._evaluate(useful_feats)
        return score

    @property
    def vocabulary(self):
        return self.features + self.ops


def _clip5(X):
    """Clip (robust-scaled) features to +/-5 to bound linear extrapolation.
    Named (not a lambda) so the pipeline stays picklable under parallel eval."""
    return np.clip(X, -5.0, 5.0)


def make_downstream_model(task, name="RF", *, n_estimators=10, random_state=0):
    """Build the model used by score_mode='downstream' (search + greedy select).

    'RF'  -> RandomForest (n_estimators trees).
    'LIN' -> robust scaled linear pipeline (median-impute -> RobustScaler ->
             clip(+/-5) -> LogisticRegression / RidgeCV). MUST mirror
             eval_quality._make_linear_pipeline so the search optimizes exactly
             the metric the eval reports. RobustScaler+clip keep heavy-tailed
             symbolic features from blowing up the linear model.
    """
    if name == "LIN":
        from sklearn.preprocessing import RobustScaler, FunctionTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import RidgeCV
        steps = [
            ('imp', SimpleImputer(strategy='median')),
            ('sc', RobustScaler()),
            ('clip', FunctionTransformer(_clip5)),
        ]
        if task == 'regression':
            steps.append(('reg', RidgeCV(alphas=np.logspace(-3, 3, 13))))
        else:
            steps.append(('clf', LogisticRegression(max_iter=2000)))
        return Pipeline(steps)
    # default: RandomForest
    if task == 'regression':
        return RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    return RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)


class SklearnEnv(Environment):
    def __init__(self, *args, n_estimators=10, scoring="f1_micro", model="RF",
                 score_mode="cv", use_cmi=False, downstream_model="RF",
                 subgroup_rule=0, **kwargs):
        """
        score_mode:
            "cv"      — feature scored by downstream sklearn CV (legacy DIFER).
            "mi"      — MI(f(X); y) using sklearn's KSG-style estimator. Used by
                        the bi-objective B-MAFE pipeline; downstream eval is
                        moved out of the search loop and run post-hoc.
            "pearson" — |Pearson(f(X), y)|. Same bi-objective composite shape as
                        'mi' (S(f;lam) = lam*U_norm + (1-lam)*(1-K_norm)), just
                        a cheaper, linear-only usefulness signal. Useful for
                        ablations against MI on numeric-target datasets.
            "fisher"  — F-statistic of the univariate linear regression
                        y ~ a + b*f(X) (Eq. 4 in the formulation). Same bi-
                        objective composite shape as 'mi' / 'pearson'. Better
                        than Pearson at distinguishing strong from very strong
                        linear signal (unbounded above) at the cost of being
                        scale-sensitive — features with f.std() == 0 score 0.

        use_cmi: when True (only meaningful with score_mode='mi'), replaces the
            raw-MI usefulness signal with residual conditional MI I(f; y | X_base).
            The residual r = y - ŷ(X_base) (regression) or 1 - p̂(y_true | X_base)
            (classification) is computed once at init via out-of-fold RF
            predictions, and every per-feature score is MI(f; r). Redundant
            features that the base feature set already explains get downweighted.
            Silently ignored under score_mode in {'pearson','fisher'} — those
            usefulness metrics are linear and the CMI residual machinery does
            not generalize the same way.
        """
        super(SklearnEnv, self).__init__(*args, **kwargs)
        self.n_estimators = n_estimators
        self.scoring = scoring
        self.model = model
        if score_mode not in ("cv", "mi", "pearson", "fisher", "downstream",
                              "split", "split_ca", "anom_auc", "anom_tail",
                              "subgroup", "subgroup_op", "rule_out", "g3out",
                              "normout", "normg3", "ifout"):
            raise ValueError(
                f"score_mode must be 'cv', 'mi', 'pearson', 'fisher', "
                f"'downstream', 'split', 'split_ca', 'anom_auc', "
                f"'anom_tail', 'subgroup', 'subgroup_op', 'rule_out', "
                f"or 'g3out', got {score_mode!r}"
            )
        self.score_mode = score_mode
        self.subgroup_rule = int(subgroup_rule or 0)
        self.downstream_model = downstream_model
        self.use_cmi = bool(use_cmi) and score_mode == "mi"
        self._residual = None  # cached y-residual when use_cmi is on
        if self.use_cmi:
            self._build_residual()
        self._cluster2 = None  # cached 2-means labels for score_mode='split_ca'
        if score_mode == "split_ca":
            self._cluster2_labels()  # eager, so parallel workers inherit it
        self._sub_gt = None  # cached rule subgroups for score_mode='subgroup*'
        if score_mode in ("subgroup", "subgroup_op"):
            self._subgroup_masks()  # eager, so parallel workers inherit it
        elif score_mode == "rule_out":
            self._outlier_masks()  # conformers vs in-rule exceptions
        elif score_mode == "g3out":
            self._g3out_masks()  # two-sided rule followers vs violators
        elif score_mode == "normout":
            self._normout_masks()  # two-rule normal vs misclassified
        elif score_mode == "normg3":
            self._normout_masks()  # same two-rule GT, balanced-g3 objective
        elif score_mode == "ifout":
            self._ifout_masks()  # frozen IForest verdict, label-free

    def _build_residual(self):
        from autolearn.feat_selection.nfo.metrics import residual_from_base
        x = self.dataset.instances
        y = np.asarray(self.dataset.labels)
        task = self.dataset.task
        X_base = x.values if hasattr(x, 'values') else np.asarray(x)
        self._residual = residual_from_base(
            X_base, y, task,
            n_estimators=self.n_estimators, cv=int(self.cv),
        )
        logger(f"[CMI] residual built: task={task} shape={self._residual.shape} "
               f"mean={float(np.mean(self._residual)):.4f} "
               f"std={float(np.std(self._residual)):.4f}")

    def mi_evaluate(self, feats):
        """Mean usefulness across all columns in feats (typically called with one).

        When use_cmi is on, the usefulness signal is MI(f; r), where r is the
        precomputed residual against the base feature set. Otherwise it is the
        raw MI(f; y).
        """
        if not feats:
            return 0.0
        if self.use_cmi:
            from autolearn.feat_selection.nfo.metrics import conditional_mutual_information
            scores = [conditional_mutual_information(col, self._residual) for col in feats]
            return float(np.mean(scores))
        from autolearn.feat_selection.nfo.metrics import mutual_information
        y = np.asarray(self.dataset.labels)
        task = self.dataset.task
        scores = [mutual_information(col, y, task) for col in feats]
        return float(np.mean(scores))

    def pearson_evaluate(self, feats):
        """Mean |Pearson(f, y)| across columns. Linear, cheap, task-agnostic.

        For classification targets we still cast y to float — Pearson is a
        valid usefulness signal against integer-coded labels (it measures
        monotone-linear association), which is what the bi-objective composite
        normalizes against Kext. It is *not* a substitute for MI when the
        relationship is non-monotone; the lam=1 endpoint of the surrogate
        will simply pick the most linearly-correlated features.
        """
        if not feats:
            return 0.0
        from autolearn.feat_selection.nfo.metrics import pearson
        y = np.asarray(self.dataset.labels, dtype=np.float64)
        scores = [pearson(col, y) for col in feats]
        return float(np.mean(scores))

    def fisher_evaluate(self, feats):
        """Mean univariate F-statistic across columns (Eq. 4).

        F = SS_reg / (SS_res / (N-2)) for the regression y ~ a + b*f. Same
        linear association measure as Pearson but unbounded above and more
        sensitive to large-N strong-signal cases. Composite normalizes via
        the eval-time pool MinMax so absolute scale doesn't break ranking
        against Kext.
        """
        if not feats:
            return 0.0
        from autolearn.feat_selection.nfo.metrics import fisher_score
        y = np.asarray(self.dataset.labels, dtype=np.float64)
        scores = [fisher_score(col, y) for col in feats]
        return float(np.mean(scores))

    def split_evaluate(self, feats):
        """Mean label-free Otsu split quality (score_mode='split').

        U(f) = between-class variance ratio of the best single-threshold
        split of f's distribution — no labels involved. The lam=1 endpoint of
        the composite searches for features whose distribution splits the
        records most cleanly in two (data-understanding objective)."""
        if not feats:
            return 0.0
        from autolearn.feat_selection.nfo.metrics import otsu_score
        return float(np.mean([otsu_score(col) for col in feats]))

    def anomaly_auc_evaluate(self, feats):
        """Mean |2*AUC-1| of each column as an anomaly score
        (score_mode='anom_auc'; labels = binary anomaly indicator, 1=anomaly)."""
        if not feats:
            return 0.0
        from autolearn.feat_selection.nfo.metrics import anomaly_auc
        y = np.asarray(self.dataset.labels)
        return float(np.mean([anomaly_auc(col, y) for col in feats]))

    def tail_evaluate(self, feats):
        """Mean label-free tail-contrast score (score_mode='anom_tail')."""
        if not feats:
            return 0.0
        from autolearn.feat_selection.nfo.metrics import tail_contrast
        return float(np.mean([tail_contrast(col) for col in feats]))

    def _cluster2_labels(self):
        """Cached 2-means assignment of the standardized base matrix.

        The label-free structure reference for score_mode='split_ca'. Computed
        once per env (eagerly at init so parallel eval workers inherit it)."""
        if getattr(self, '_cluster2', None) is None:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            X = self.dataset.instances.to_numpy(dtype=np.float64)
            X = np.where(np.isfinite(X), X, np.nan)
            med = np.nanmedian(X, axis=0)
            idx = np.where(np.isnan(X))
            X[idx] = np.take(med, idx[1])
            Xs = StandardScaler().fit_transform(X)
            self._cluster2 = KMeans(n_clusters=2, n_init=10,
                                    random_state=0).fit_predict(Xs)
            logger(f"[split_ca] cached 2-means of base X: "
                   f"sizes={np.bincount(self._cluster2).tolist()}")
        return self._cluster2

    def _subgroup_masks(self):
        """Cached opposing-rule subgroup masks (score_mode='subgroup').

        Ground truth per paper_drafts/subgroup_plan.md: a single depth-3
        decision tree on (X, y), leaf rules scored by WRAcc, the best rule
        toward each class defines subgroups A and B. Computed eagerly at init
        so parallel eval workers inherit it; deterministic (seed 0)."""
        if getattr(self, '_sub_gt', None) is None:
            from autolearn.feat_selection.nfo.metrics import subgroup_ground_truth
            X = self.dataset.instances.to_numpy(dtype=np.float64)
            y = np.asarray(self.dataset.labels)
            gt = subgroup_ground_truth(
                X, y, opposite=(self.score_mode == 'subgroup_op'),
                rule_rank=getattr(self, 'subgroup_rule', 0))
            self._sub_gt = gt
            logger(f"[subgroup] rule_rank={getattr(self, 'subgroup_rule', 0)}")
            logger(f"[subgroup] rule A (WRAcc={gt['wracc_a']:+.4f}, "
                   f"n={int(gt['mask_a'].sum())}): {gt['rule_a']}")
            logger(f"[subgroup] rule B (WRAcc={gt['wracc_b']:+.4f}, "
                   f"n={int(gt['mask_b'].sum())}): {gt['rule_b']}")
        return self._sub_gt

    def _outlier_masks(self):
        """Cached conformer/exception masks (score_mode='rule_out').

        Ground truth: the rule_rank-th valid |WRAcc| rule of the same
        depth-3 tree; conformers = in-rule rows with the rule's majority
        label (mask_a), outliers = in-rule exceptions (mask_b). The dict
        shape matches _subgroup_masks so subgroup_evaluate is reused."""
        if getattr(self, '_sub_gt', None) is None:
            from autolearn.feat_selection.nfo.metrics import outlier_ground_truth
            X = self.dataset.instances.to_numpy(dtype=np.float64)
            y = np.asarray(self.dataset.labels)
            gt = outlier_ground_truth(
                X, y, rule_rank=getattr(self, 'subgroup_rule', 0))
            self._sub_gt = gt
            logger(f"[rule_out] rule_rank={getattr(self, 'subgroup_rule', 0)}"
                   f" of {gt['n_valid']} valid rules")
            logger(f"[rule_out] conformers (n={int(gt['mask_a'].sum())}): "
                   f"{gt['rule_a']}")
            logger(f"[rule_out] exceptions (n={int(gt['mask_b'].sum())}): "
                   f"{gt['rule_b']}")
        return self._sub_gt

    def _g3out_masks(self):
        """Cached two-sided conformity masks (score_mode='g3out').

        Ground truth: the rule_rank-th valid |WRAcc| rule and its opposite
        counterpart; conformers = rows following their own side's majority
        label (mask_a), violators = minority-label rows on either side
        (mask_b). Dict shape matches the other rule-based modes."""
        if getattr(self, '_sub_gt', None) is None:
            from autolearn.feat_selection.nfo.metrics import ruleconf_ground_truth
            X = self.dataset.instances.to_numpy(dtype=np.float64)
            y = np.asarray(self.dataset.labels)
            gt = ruleconf_ground_truth(
                X, y, rule_rank=getattr(self, 'subgroup_rule', 0))
            self._sub_gt = gt
            logger(f"[g3out] rule_rank={getattr(self, 'subgroup_rule', 0)}"
                   f" of {gt['n_valid']} valid rule pairs")
            logger(f"[g3out] rule: {gt['rule']} (maj A={gt['maj_a']}, "
                   f"B={gt['maj_b']})")
            logger(f"[g3out] conformers n={int(gt['mask_a'].sum())}, "
                   f"violators n={int(gt['mask_b'].sum())}")
        return self._sub_gt

    def g3out_evaluate(self, feats):
        """Mean class-balanced (1 - g3) threshold separability between the
        frozen rule followers and violators (score_mode='g3out')."""
        if not feats:
            return 0.0
        from autolearn.feat_selection.nfo.metrics import g3_balanced
        gt = self._g3out_masks()
        return float(np.mean([g3_balanced(col, gt['mask_a'], gt['mask_b'])
                              for col in feats]))

    def _normout_masks(self):
        """Cached normal/outlier masks (score_mode='normout').

        Ground truth: the rule_rank-th valid pair of discovered tree
        rules, one predicting each class; normal = covered and correctly
        classified by either rule (mask_a), outlier = covered but
        misclassified (mask_b). Objective = Cohen's d via
        subgroup_evaluate (maximize normal-outlier distance)."""
        if getattr(self, '_sub_gt', None) is None:
            from autolearn.feat_selection.nfo.metrics import normout_ground_truth
            X = self.dataset.instances.to_numpy(dtype=np.float64)
            y = np.asarray(self.dataset.labels)
            gt = normout_ground_truth(
                X, y, rule_rank=getattr(self, 'subgroup_rule', 0))
            self._sub_gt = gt
            logger(f"[normout] rule_rank={getattr(self, 'subgroup_rule', 0)}"
                   f" of {gt['n_valid']} valid pairs")
            logger(f"[normout] rule A: {gt['rule_a']}")
            logger(f"[normout] rule B: {gt['rule_b']}")
            logger(f"[normout] normal n={int(gt['mask_a'].sum())}, "
                   f"outliers n={int(gt['mask_b'].sum())}")
        return self._sub_gt

    def _ifout_masks(self):
        """Cached inlier/outlier masks (score_mode='ifout', label-free).

        Ground truth: a frozen IsolationForest fitted on the base matrix;
        mask_a = inliers, mask_b = flagged outliers (bottom 10% scores).
        Objective = Cohen's d via subgroup_evaluate: one readable
        expression reproducing the detector's verdict."""
        if getattr(self, '_sub_gt', None) is None:
            from autolearn.feat_selection.nfo.metrics import ifout_ground_truth
            X = self.dataset.instances.to_numpy(dtype=np.float64)
            # --subgroup_rule doubles as the contamination selector here:
            # 0 -> 10% (default), 1 -> 5%, 2 -> 15%.
            alpha = {0: 0.10, 1: 0.05, 2: 0.15}.get(
                int(getattr(self, 'subgroup_rule', 0) or 0), 0.10)
            gt = ifout_ground_truth(X, contamination=alpha)
            self._sub_gt = gt
            logger(f"[ifout] frozen IForest verdict: inliers "
                   f"n={int(gt['mask_a'].sum())}, outliers "
                   f"n={int(gt['mask_b'].sum())} "
                   f"(contamination={gt['contamination']:.0%})")
        return self._sub_gt

    def subgroup_evaluate(self, feats):
        """Mean standardized subgroup distance U_sub(f) = Cohen's d between
        the frozen rule-defined subgroups in the feature's values."""
        if not feats:
            return 0.0
        from autolearn.feat_selection.nfo.metrics import cohens_d
        gt = self._subgroup_masks()
        return float(np.mean([cohens_d(col, gt['mask_a'], gt['mask_b'])
                              for col in feats]))

    def split_ca_evaluate(self, feats):
        """Mean cluster-agreement split quality (score_mode='split_ca').

        U(f) = ARI between the column's best-threshold (Otsu) split and the
        cached 2-means clustering of the standardized base matrix — label-free,
        and not gameable by near-binary features (the pilot showed plain Otsu
        is: a two-valued column scores 1.0 regardless of meaning). A feature
        scores highly only if its one-dimensional split reproduces the
        dataset's dominant multivariate structure."""
        if not feats:
            return 0.0
        from sklearn.metrics import adjusted_rand_score
        from autolearn.feat_selection.nfo.metrics import otsu_threshold, _clean
        cl = self._cluster2_labels()
        scores = []
        for col in feats:
            f = _clean(col)
            t = otsu_threshold(f)
            if not np.isfinite(t):
                scores.append(0.0)
                continue
            grp = (f > t).astype(int)
            if grp.min() == grp.max():
                scores.append(0.0)
                continue
            scores.append(max(0.0, float(adjusted_rand_score(cl, grp))))
        return float(np.mean(scores))

    def sklearn_evaluate(self, feats, row_idx=None):
        """Downstream CV of [base + feats]. ``row_idx`` (optional int array)
        restricts the CV to a row subsample — used by the LSBO rerank cap so
        the heuristic stays O(1) in dataset size."""
        x = copy.deepcopy(self.dataset.instances)
        y = copy.deepcopy(self.dataset.labels)
        x = self.add_features(x, feats)
        x.replace([np.inf, -np.inf], np.nan, inplace=True)
        x.fillna(-1, inplace=True)
        x.columns = x.columns.astype(str)
        if row_idx is not None:
            x = x.iloc[row_idx].reset_index(drop=True)
            y = np.asarray(y)[np.asarray(row_idx)]

        if self.dataset.task == 'regression':
            if self.model == 'LR':
                model = Lasso()
            elif self.model == 'SVM':
                model = LinearSVR()
            elif self.model == "LGB":
                model = LGBMRegressor()
            elif self.model == "XGB":
                model = XGBRegressor()
            else:
                model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=0)
            score = cross_val_score(model, x, y, scoring=make_scorer(r2_score), cv=int(self.cv)).mean()
        elif self.dataset.task == 'classification':
            if self.model == 'LR':
                model = LogisticRegression()
            elif self.model == 'SVM':
                model = LinearSVC()
            elif self.model == 'LGB':
                model = LGBMClassifier()
            elif self.model == "XGB":
                model = XGBClassifier()
            else:
                model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=0)
            score = cross_val_score(model, x, y, scoring=self.scoring, cv=int(self.cv)).mean()
        else:
            score = -1
        return score

    def downstream_evaluate(self, feats):
        """Downstream CV usefulness of [base ∪ feats] (score_mode='downstream').

        The model is ``self.downstream_model`` ('RF' or 'LIN'), so the
        bi-objective B-MAFE search optimizes the real downstream performance of
        whichever model the evaluation reports — instead of a cheap proxy (MI /
        Pearson / Fisher). Plugged into the same composite machinery: U(f) = CV
        of [base ∪ f], and S(f;λ) = λ·U_norm + (1-λ)·(1-Kext_norm). At λ=1 the
        composite reduces to the pool-normalized CV score.

        NOTE: each call trains a 5-fold model, so this is ~1000x costlier per
        feature than the MI/Pearson/Fisher proxies. The initial pool size
        (config.random_set_size) is the dominant evaluation-budget knob.

        For the 'LIN' pipeline we leave inf->NaN in place (its imputer handles
        NaN on the standardized scale); for 'RF' we keep the legacy -1 fill.
        """
        x = copy.deepcopy(self.dataset.instances)
        y = copy.deepcopy(self.dataset.labels)
        x = self.add_features(x, feats)
        x.replace([np.inf, -np.inf], np.nan, inplace=True)
        if self.downstream_model != 'LIN':
            x.fillna(-1, inplace=True)
        x.columns = x.columns.astype(str)
        task = self.dataset.task
        if task not in ('regression', 'classification'):
            return -1
        model = make_downstream_model(task, self.downstream_model,
                                      n_estimators=self.n_estimators)
        scorer = make_scorer(r2_score) if task == 'regression' else self.scoring
        return cross_val_score(model, x, y, scoring=scorer, cv=int(self.cv)).mean()

    def _evaluate(self, feats):
        if self.score_mode == "mi":
            return self.mi_evaluate(feats)
        if self.score_mode == "pearson":
            return self.pearson_evaluate(feats)
        if self.score_mode == "fisher":
            return self.fisher_evaluate(feats)
        if self.score_mode == "downstream":
            return self.downstream_evaluate(feats)
        if self.score_mode == "split":
            return self.split_evaluate(feats)
        if self.score_mode == "split_ca":
            return self.split_ca_evaluate(feats)
        if self.score_mode in ("subgroup", "subgroup_op", "rule_out",
                               "normout", "ifout"):
            return self.subgroup_evaluate(feats)
        if self.score_mode in ("g3out", "normg3"):
            # normg3: _sub_gt was cached by _normout_masks at init, so
            # g3out_evaluate scores the normout masks with balanced 1-g3.
            return self.g3out_evaluate(feats)
        if self.score_mode == "anom_auc":
            return self.anomaly_auc_evaluate(feats)
        if self.score_mode == "anom_tail":
            return self.tail_evaluate(feats)
        return self.sklearn_evaluate(feats)


