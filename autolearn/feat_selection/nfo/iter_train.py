import argparse
import torch
import random
import os
import numpy as np
from sklearn.model_selection import train_test_split
import sys

sys.path.extend([
    ".",
])

from autolearn.feat_selection.nfo.feat_tree import generate_tree_from_str, FeatureGenerator
from autolearn.feat_selection.nfo.tokenizer import Tokenizer
from autolearn.feat_selection.nfo.controller import NFOController, Mode, load_nfo
from autolearn.utils import Config, multi_train, timeit, log, get_time_budget, set_time_budget
from autolearn.utils.train import pairwise_hinge_loss
from autolearn.feat_selection.env import SklearnEnv
from autolearn.feat_selection.dataset import MixDataset

is_cuda = None
device = None

# score_modes that drive the bi-objective B-MAFE composite path
# S(f;λ) = λ·U_norm + (1-λ)·(1-Kext_norm). They differ only in the usefulness
# signal U: 'mi'/'pearson'/'fisher' are cheap per-feature proxies; 'downstream'
# is the real downstream RF CV (use with λ=1 to optimize downstream perf and
# ignore explainability). All share the same surrogate target, pool ranking,
# and selected_topk manifest writer.
COMPOSITE_MODES = ('mi', 'pearson', 'fisher', 'downstream',
                   'split', 'split_ca', 'anom_auc', 'anom_tail',
                   'subgroup', 'subgroup_op', 'rule_out', 'g3out', 'normout',
                   'normg3', 'ifout')


def parse_args():
    parser = argparse.ArgumentParser(description='NFO')
    parser.add_argument('--data', type=str, help='dataset name')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--cuda', type=str, default="0", help='which gpu to use')
    parser.add_argument(
        '--parallel_eval', action='store_true', help="parallel evaluation when evaluate new features"
    )
    parser.add_argument(
        '--ckp_path', type=str, default=None, help="checkpoint path when training"
    )
    parser.add_argument('--cv', type=int, default=5, help='evaluate cv')
    parser.add_argument('--hyper_config', type=str, default='default')
    parser.add_argument('--eval_model', type=str, default='RF')
    parser.add_argument('--feat_pool', type=str, default=None)
    parser.add_argument('--new_feat', type=int, default=32)
    parser.add_argument('--top_feat', type=int, default=96)
    parser.add_argument('--cur_iter', type=int, default=0)
    parser.add_argument('--total_iter', type=int, default=10)
    parser.add_argument('--n_estimators', type=int, default=10)
    parser.add_argument('--c_scoring', type=str, default="f1_micro")
    parser.add_argument('--scratch', type=bool, default=True)
    parser.add_argument(
        '--method', type=str, default='', choices=['', 'lsbo', 'infer', 'vi'],
        help="feature-generation method override; 'lsbo'/'infer'/'vi' force the choice; '' uses config.use_lsbo / config.use_vi"
    )
    parser.add_argument(
        '--score_mode', type=str, default='cv',
        choices=['cv', 'mi', 'pearson', 'fisher', 'downstream',
                 'split', 'split_ca', 'anom_auc', 'anom_tail',
                 'subgroup', 'subgroup_op', 'rule_out', 'g3out', 'normout',
                 'normg3', 'ifout'],
        help="'cv' = legacy DIFER (downstream sklearn CV score). "
             "'mi' = bi-objective B-MAFE (model-agnostic mutual information). "
             "'pearson' = bi-objective B-MAFE with |Pearson(f, y)|. "
             "'fisher' = bi-objective B-MAFE with the F-statistic of y ~ a + b*f "
             "(Eq. 4). 'downstream' = bi-objective B-MAFE whose usefulness U(f) "
             "is the real downstream RF CV of [base + f] (use with --lam 1 to "
             "optimize downstream performance directly, ignoring explainability). "
             "All of mi/pearson/fisher/downstream share the composite shape "
             "S(f;lam) = lam*U_norm + (1-lam)*(1-Kext_norm); the first three move "
             "downstream eval out of the search loop, 'downstream' puts it back in."
    )
    parser.add_argument(
        '--lam', type=float, default=1.0,
        help="Trade-off in S(f;lam) = lam*U + (1-lam)*E. lam=1 = pure usefulness, "
             "lam=0 = pure simplicity. Only used when score_mode in "
             "('mi', 'pearson', 'fisher', 'downstream')."
    )
    parser.add_argument(
        '--max_order', type=int, default=None,
        help="Override config.max_order at runtime. When set, the env is "
             "constructed with this depth budget instead of the value baked "
             "into the hyper-config JSON. Used by the efficiency sweep to "
             "vary expression length without authoring a per-depth config."
    )
    parser.add_argument(
        '--random_set_size', type=int, default=None,
        help="Override config.random_set_size at runtime. Controls the size of "
             "the initial random expression pool the FeatureGenerator builds "
             "before surrogate training. Used by the efficiency sweep to vary "
             "the initial-sample-size axis without authoring per-size configs."
    )
    parser.add_argument(
        '--patience', type=int, default=None,
        help="Override config.patience (validation-loss early-stop window). "
             "Pass a large value (e.g. 999999) to disable early stopping for "
             "the efficiency sweep so the predictor trains the full --num_epoch "
             "budget every cell — makes the per-epoch sequence-length cost "
             "dominant and reveals the max_order scaling that early-stop hides."
    )
    parser.add_argument(
        '--num_epoch', type=int, default=None,
        help="Override config.num_epoch (max training epochs for the surrogate). "
             "Paired with --patience for efficiency measurements: fixed-epoch "
             "training gives a clean, reproducible compute budget per cell."
    )
    parser.add_argument(
        '--num_ops_only', action='store_true',
        help="Force the operator search space to numerical operators only "
             "(+, -, *, /, log, sqrt_abs, min_max, reciprocal, mod_column), "
             "even when the dataset has categorical columns. Used by the "
             "Pearson sweep so the operator space matches a linear-correlation "
             "usefulness metric. Overrides config.use_cat_space."
    )
    parser.add_argument(
        '--select_k', type=int, default=None,
        help="Number of features written to the selected_topk manifest "
             "(final top-k). Default: max(new_feat, n_base_features). "
             "Selection is a rank-and-take over the evaluated pool, so this "
             "knob is expected to have no runtime impact -- exposed so the "
             "scalability study can measure that claim rather than assume it."
    )
    parser.add_argument(
        '--ops', type=str, default=None,
        help="Comma-separated operator-name whitelist for the |O| vocabulary "
             "ablation (e.g. '+,-,*,division,log'). Applied after "
             "--num_ops_only, so names must exist in the active op space. "
             "Shrinks the tokenizer vocab and Kext's log2|O| term "
             "consistently. Use fresh --feat_pool/--ckp_path dirs: caches "
             "built with a different vocabulary are not loadable."
    )
    parser.add_argument(
        '--subgroup_rule', type=int, default=0,
        help="Rank of the WRAcc rule used as subgroup A under score_mode="
             "'subgroup'/'subgroup_op' (0 = best rule, 1 = second best, ...). "
             "Used by the multi-rule subgroup experiments; the opposite-rule "
             "counterpart B is derived from the selected rule."
    )
    parser.add_argument(
        '--downstream_model', type=str, default=None,
        choices=['RF', 'LIN'],
        help="model optimized by score_mode='downstream' (pool scoring + greedy "
             "select). 'RF' (default) = RandomForest; 'LIN' = robust scaled "
             "linear (impute+RobustScaler+clip+LogReg/RidgeCV), matching "
             "eval_quality's LIN model so the search optimizes exactly what the "
             "linear eval reports. Falls back to config.downstream_model then 'RF'."
    )
    parser.add_argument(
        '--greedy_selector', type=str, default=None, choices=['forward', 'sffs'],
        help="set-level selector for score_mode='downstream': 'forward' "
             "(sequential forward, best-prefix early stop) or 'sffs' (floating, "
             "with backward removals). Falls back to config.downstream_greedy."
             "selector then 'forward'. SFFS is stronger but can overfit the "
             "held-out criterion on small datasets."
    )
    parser.add_argument(
        '--method_label', type=str, default='',
        help="Label written into the manifest's 'method' field, overriding "
             "the default (which mirrors --method). Lets sister experiments "
             "that share a --method but differ in score_mode / op-set / etc. "
             "(e.g. lsbo + MI vs. lsbo + Pearson + MDL) get distinct labels "
             "in eval_quality plots and row IDs."
    )
    args = parser.parse_args()
    return args


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def set_cuda(cuda):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda
    global is_cuda, device
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda', 0) if is_cuda else torch.device('cpu')


def get_config(path):
    config = Config(path)
    return config


def _downstream_rerank_lsbo(new_feats, env, *, top_n=0, cap_rows=10000):
    """Rerank LSBO candidates by their individual downstream CV score.

    For each candidate f, build the constructed column and call
    ``env.sklearn_evaluate([col])`` directly (NOT ``env.eval_set``, which
    would dispatch to ``mi_evaluate`` / CMI under score_mode='mi' and
    return the search-time metric instead of real downstream CV). Sort
    descending. Features that fail to parse / evaluate keep their original
    relative position at the back.

    ``top_n`` truncates the input list to its first N entries before
    scoring (saves CV time when LSBO returns many candidates). 0 = no cap.

    ``cap_rows`` bounds the CV cost in N: when the dataset has more rows,
    the rerank CV runs on a fixed seeded row subsample, making this optional
    heuristic O(1) in dataset size (a ranking signal does not need every
    row; uncapped, RF CV at N=10^6 costs hours per iteration and would put
    a downstream model back on the scaling path). 0 disables the cap.
    """
    if not new_feats:
        return new_feats
    if top_n and top_n > 0:
        head, tail = list(new_feats[:top_n]), list(new_feats[top_n:])
    else:
        head, tail = list(new_feats), []

    n_rows = len(env.dataset.labels)
    row_idx = None
    if cap_rows and n_rows > cap_rows:
        row_idx = np.random.default_rng(0).choice(n_rows, cap_rows,
                                                  replace=False)
        log(f"[LSBO] downstream rerank capped: CV on {cap_rows}/{n_rows} "
            f"seeded subsample rows")

    scored, unscored = [], []
    t0 = __import__('time').time()
    for f in head:
        try:
            tree = generate_tree_from_str([f], env)[0]
            if tree is None:
                raise ValueError("unparseable")
            col = env.construct_feature(tree)
            cv = float(env.sklearn_evaluate([col], row_idx=row_idx))
        except Exception:
            unscored.append(f)
            continue
        if not np.isfinite(cv):
            unscored.append(f)
            continue
        scored.append((cv, f))

    scored.sort(key=lambda kv: kv[0], reverse=True)
    elapsed = __import__('time').time() - t0
    if scored:
        log(f"[LSBO] downstream rerank: scored {len(scored)}/{len(head)} "
            f"in {elapsed:.1f}s; CV [{scored[0][0]:.4f} -> {scored[-1][0]:.4f}]")
    else:
        log(f"[LSBO] downstream rerank: 0/{len(head)} scored "
            f"({elapsed:.1f}s); keeping original order")
        return new_feats
    return [f for _, f in scored] + unscored + tail


@timeit
def main(args):
    log(f"Current iteration {args.cur_iter} for dataset {args.data}")
    # 1. prepare environments
    set_cuda(args.cuda)
    global is_cuda, device
    set_random_seed(args.seed)
    file_path = os.path.abspath(os.path.dirname(__file__))
    global_config = get_config(f"{file_path}/config/{args.hyper_config}.json")
    config = global_config
    # Normalize method dispatch — accept CLI override, fall back to config flags.
    if args.method == 'vi':
        method = 'vi'
    elif args.method == 'lsbo':
        method = 'lsbo'
    elif args.method == 'infer':
        method = 'infer'
    elif config.use_vi:
        method = 'vi'
    elif config.use_lsbo:
        method = 'lsbo'
    else:
        method = 'infer'
    config['use_lsbo'] = (method == 'lsbo')
    log(f"use device {device}")
    log(f"feature-generation method: {method.upper()}"
        + (" (legacy DIFER)" if method == 'infer' else ""))

    force_num_ops = bool(args.num_ops_only) or bool(
        config.num_ops_only if config.num_ops_only is not None else False
    )
    downstream_model = _resolve_downstream_model(args, config)
    if args.score_mode == 'downstream':
        log(f"[B-MAFE] score_mode=downstream optimizing downstream_model={downstream_model}")
    env = SklearnEnv(
        args.data, args.max_order if args.max_order is not None else config.max_order,
        model=args.eval_model,
        cv=args.cv, parallel=args.parallel_eval,
        n_estimators=args.n_estimators, scoring=args.c_scoring,
        score_mode=args.score_mode,
        downstream_model=downstream_model,
        use_cmi=bool(config.use_cmi) if config.use_cmi is not None else False,
        use_cat_space=bool(config.use_cat_space) if config.use_cat_space is not None else True,
        force_num_ops=force_num_ops,
        ops_subset=args.ops.split(',') if args.ops else None,
        subgroup_rule=getattr(args, 'subgroup_rule', 0),
    )
    if args.ops:
        log(f"[B-MAFE] ops subset ON — |O|={len(env.ops)}: {env.ops}")
    if force_num_ops:
        log(f"[B-MAFE] num_ops_only ON — operator space = numerical only "
            f"({len(env.ops)} ops)")
    if args.score_mode in COMPOSITE_MODES:
        if args.score_mode == 'pearson':
            usefulness_name = '|Pearson(f; y)|'
        elif args.score_mode == 'fisher':
            usefulness_name = 'F-stat(y ~ a + b*f)'
        elif args.score_mode == 'downstream':
            usefulness_name = 'downstream RF CV(base + f)'
        elif args.score_mode == 'split':
            usefulness_name = 'Otsu split quality (label-free)'
        elif args.score_mode == 'split_ca':
            usefulness_name = 'cluster-agreement split ARI (label-free)'
        elif args.score_mode == 'subgroup':
            usefulness_name = "Cohen's d between rule subgroups"
        elif args.score_mode == 'subgroup_op':
            usefulness_name = "Cohen's d, opposite-rule subgroups"
        elif args.score_mode == 'rule_out':
            usefulness_name = "Cohen's d, rule conformers vs exceptions"
        elif args.score_mode == 'g3out':
            usefulness_name = "balanced 1-g3, rule followers vs violators"
        elif args.score_mode == 'normout':
            usefulness_name = "Cohen's d, rule-normal vs misclassified"
        elif args.score_mode == 'normg3':
            usefulness_name = "balanced 1-g3, rule-normal vs misclassified"
        elif args.score_mode == 'ifout':
            usefulness_name = "Cohen's d, IForest inliers vs outliers"
        elif args.score_mode == 'anom_auc':
            usefulness_name = '|2*AUC(f; anomaly)-1|'
        elif args.score_mode == 'anom_tail':
            usefulness_name = 'tail contrast (label-free)'
        else:
            usefulness_name = 'CMI(f; r)' if env.use_cmi else 'MI(f; y)'
        log(f"[B-MAFE] score_mode={args.score_mode}  lam={args.lam}  usefulness={usefulness_name}  "
            f"(surrogate target = lam*U_norm + (1-lam)*(1-Kext_norm))")
    set_time_budget(env.dataset.time_budget)
    time_budget = get_time_budget()

    # 2. initialize tokenizer & controller
    tokenizer = Tokenizer(env.vocabulary, env.max_length)
    encoder_kind = (config.encoder_kind or 'gat').lower()
    vae_flag = bool(config.vae) if config.vae is not None else False
    log(f"controller encoder_kind = {encoder_kind}  vae = {vae_flag}")
    nfo = NFOController(
        tokenizer.vocab_size,
        tokenizer.max_length,
        tokenizer.padding_idx,
        tokenizer.start_idx,
        tokenizer=tokenizer,
        op_arity=env.op_arity,
        encoder_kind=encoder_kind,
        vae=vae_flag,
    )
    if args.scratch:
        nfo, use_iter = nfo, None
    else:
        nfo, use_iter = load_nfo(nfo, args.ckp_path)
    nfo.to(device)

    # 3. train controller when explore & exploit
    #   - use RandomGenerator generate features and evaluate them, finally get dataset F(feat, score)
    #   - Data Augmentation: each feature tree may have many post order representations
    effective_random_set_size = (
        args.random_set_size if args.random_set_size is not None
        else config.random_set_size
    )
    feature_generator = FeatureGenerator(
        env,
        cur_iter=args.cur_iter,
        feature_pool=args.feat_pool,
        population_size=effective_random_set_size,
        ckp_path=args.ckp_path,
        use_iter=use_iter
    )
    features, scores = feature_generator.dataset
    seqs = tokenizer.transform(features)

    sample_num = int(min(10, effective_random_set_size))
    args.top_feat = min(args.top_feat, len(feature_generator))

    # When score_mode in {'mi','pearson'}, the pool stores the raw usefulness
    # signal (after FeatureGenerator's MinMax scaling). Replace the surrogate
    # target with the bi-objective composite S(f;lam) = lam*U_norm + (1-lam)*
    # (1-Kext_norm) so the predictor learns the Pareto front for this lam.
    if args.score_mode in COMPOSITE_MODES:
        from autolearn.feat_selection.nfo.metrics import composite_from_pool, kext as kext_fn
        n_base = len(env.features)
        n_ops = len(env.ops)
        # Recover raw (pre-scaling) usefulness values; scaler was fit on [baseline, max].
        raw_u = feature_generator.scaler.inverse_transform(
            np.asarray(scores).reshape(-1, 1)
        ).reshape(-1)
        trees = generate_tree_from_str(features, env)
        kext_scores = np.array([
            kext_fn(t, n_base, n_ops) if t is not None else float('nan')
            for t in trees
        ], dtype=np.float64)
        composite = composite_from_pool(raw_u, kext_scores, args.lam)
        # Surrogate is bounded by tanh ∈ (-1, 1); composite is in [0, 1].
        scores = composite.astype(np.float64)
        u_label = {'pearson': 'Pearson', 'fisher': 'Fisher',
                   'downstream': 'RF-CV', 'split': 'Otsu',
                   'split_ca': 'ClusterARI', 'anom_auc': 'AnomAUC',
                   'anom_tail': 'TailContrast',
                   'subgroup': 'SubgroupD',
                   'subgroup_op': 'SubgroupD-Op',
                   'rule_out': 'RuleOut',
                   'g3out': 'G3Out',
                   'normout': 'NormOut',
                   'normg3': 'NormG3',
                   'ifout': 'IFOut'}.get(args.score_mode, 'MI')
        log(f"[B-MAFE] composite target stats: mean={scores.mean():.4f} "
            f"min={scores.min():.4f} max={scores.max():.4f} "
            f"({u_label} raw mean={raw_u.mean():.4f}, Kext mean={np.nanmean(kext_scores):.4f})")

    x_train, x_valid, y_train, y_valid = train_test_split(seqs, scores, test_size=0.2, shuffle=True)
    train_set = MixDataset(x_train, y_train)
    valid_set = MixDataset(x_valid, y_valid)
    # mix_set = MixDataset(seqs, scores, seqs, seqs)
    if args.scratch:
        log(f"features: {features}", level='debug')

    #   - use F to train encoder-predictor-predictor
    score_loss_name = (config.score_loss or 'mse').lower()
    if score_loss_name == 'rank':
        score_loss_fn = pairwise_hinge_loss
    elif score_loss_name == 'mse':
        score_loss_fn = torch.nn.functional.mse_loss
    else:
        raise ValueError(f"unknown score_loss {score_loss_name!r} (expected 'mse' or 'rank')")
    log(f"predictor training loss: {score_loss_name}")

    if args.scratch:
        nfo.set_mode(Mode.EPD)
        nfo.train()
        optimizer = torch.optim.Adam(
            nfo.parameters(), lr=config.lr,
            weight_decay=config.weight_decay
        )
        kl_beta_val = float(config.kl_beta) if config.kl_beta is not None else 0.0
        if vae_flag and kl_beta_val > 0.0:
            log(f"[VAE] training with KL beta = {kl_beta_val}")
        eff_patience = args.patience if args.patience is not None else config.patience
        eff_num_epoch = args.num_epoch if args.num_epoch is not None else config.num_epoch
        if args.patience is not None or args.num_epoch is not None:
            log(f"[override] patience={eff_patience} num_epoch={eff_num_epoch}")
        multi_train(
            train_set, nfo, optimizer,
            score_loss_fn, torch.nn.functional.nll_loss,
            device,
            trade_off=config.trade_off,
            trade_off_epoch=config.trade_off_epoch,
            ckp_path=args.ckp_path,
            patience=eff_patience,
            valid_set=valid_set,
            epochs=eff_num_epoch, batch_size=config.batch_size,
            time_budget=time_budget,
            cur_iter=args.cur_iter,
            kl_beta=kl_beta_val,
        )

    if args.scratch:
        # - test encoder-predictor
        nfo.set_mode(Mode.EP)
        nfo.eval()
        scores_hat = nfo(torch.tensor(seqs, dtype=torch.long, device=device)).detach().cpu().numpy()
        log(f"original feature scores:\n{scores[: sample_num]}")
        log(f"predict feature scores: {scores_hat.reshape(-1).tolist()[: sample_num]}")

        # --- surrogate-quality dump (Sec. IV-E). No-op unless SURR_DUMP is set
        # to an output CSV path. Records true vs predicted surrogate score on
        # the held-out valid split (and train) so we can report R^2 / Spearman.
        # At score_mode=mi with lam=1 the target is pure usefulness U; with
        # lam=0 it is pure explainability E=(1-Kext_norm).
        _surr_dump = os.environ.get('SURR_DUMP')
        if _surr_dump:
            import pandas as _pd

            def _pred(_seqs):
                if len(_seqs) == 0:
                    return np.empty(0)
                t = torch.tensor(_seqs, dtype=torch.long, device=device)
                return nfo(t).detach().cpu().numpy().reshape(-1)

            _rows = []
            for _split, _xs, _ys in (('train', x_train, y_train),
                                     ('valid', x_valid, y_valid)):
                _ph = _pred(_xs)
                for _ti, _pi in zip(np.asarray(_ys).reshape(-1), _ph):
                    _rows.append({'split': _split, 'true': float(_ti),
                                  'pred': float(_pi),
                                  'lam': args.lam, 'score_mode': args.score_mode,
                                  'dataset': args.data, 'cur_iter': args.cur_iter})
            os.makedirs(os.path.dirname(_surr_dump) or '.', exist_ok=True)
            _hdr = not os.path.exists(_surr_dump)
            _pd.DataFrame(_rows).to_csv(_surr_dump, mode='a', index=False,
                                        header=_hdr)
            log(f"[SURR_DUMP] appended {len(_rows)} rows -> {_surr_dump}")

        #   - test encoder-decoder
        nfo.set_mode(Mode.ED)
        nfo.eval()
        _, decoded_seqs = nfo(torch.tensor(seqs, dtype=torch.long, device=device))
        decoded_feats = tokenizer.inverse_transform(decoded_seqs)
        log(f"original features:\n{features[:sample_num]}")
        log(f"decode features to:\n{decoded_feats[:sample_num]}")

    # 4. generate good features by NFO
    log(f"original features generated randomly:\n{features}", level='debug')
    args.top_feat = min(args.top_feat, len(feature_generator.features))

    if method == 'vi':
        from autolearn.feat_selection.nfo.amortized_vi import amortized_vi_search
        vi_cfg = config.vi or {}
        nfo.eval()
        # Anchor selection should match the surrogate's training target.
        # Under MI mode, surrogate predicts composite S(f;lam); use composite
        # for top-k anchor selection so anchors live where the surrogate is high.
        anchor_scores = scores if args.score_mode in COMPOSITE_MODES else feature_generator.scores
        new_feats = amortized_vi_search(
            nfo, tokenizer, device,
            pool_features=feature_generator.features,
            pool_scores=anchor_scores,
            padding_idx=tokenizer.padding_idx,
            dim=vi_cfg.get('dim', 512),
            hidden=vi_cfg.get('hidden', 256),
            n_layers=vi_cfg.get('n_layers', 6),
            tau=vi_cfg.get('tau', 0.1),
            beta=vi_cfg.get('beta', 1e-2),
            n_iters=vi_cfg.get('n_iters', 300),
            batch_size=vi_cfg.get('batch_size', 32),
            n_per_anchor_train=vi_cfg.get('n_per_anchor_train', 4),
            n_per_anchor_inf=vi_cfg.get('n_per_anchor_inf', 8),
            top_k_seed=vi_cfg.get('top_k_seed', 16),
            lr=vi_cfg.get('lr', 1e-3),
        )
        before_count = feature_generator.new_feat
        is_changed = feature_generator.append_new(new_feats)
        accepted = feature_generator.new_feat - before_count
        log(f"[VI] append_new: {accepted}/{len(new_feats)} accepted "
            f"({int(is_changed.sum())} unique+valid+novel). "
            f"Rejected = duplicates of pool/new_feats or invalid syntax.")
    elif method == 'lsbo':
        from autolearn.feat_selection.nfo.lsbo import lsbo_search
        lsbo_cfg = config.lsbo or {}
        nfo.eval()
        new_feats = lsbo_search(
            nfo, tokenizer, device,
            pool_features=feature_generator.features,
            pool_scores=feature_generator.scores,
            padding_idx=tokenizer.padding_idx,
            K=lsbo_cfg.get('K', 16),
            n_cands=lsbo_cfg.get('n_cands', max(args.new_feat, 32)),
            n_steps=lsbo_cfg.get('n_steps', 50),
            eta=lsbo_cfg.get('eta', 1e-3),
            xi=lsbo_cfg.get('xi', 0.01),
            r_max=lsbo_cfg.get('r_max', 2.0),
            top_k_seed=lsbo_cfg.get('top_k_seed', 8),
            sigma_noise_grid=lsbo_cfg.get('sigma_noise_grid', [0.05, 0.1, 0.2, 0.4]),
            starts_per_combo=lsbo_cfg.get('starts_per_combo', 3),
            gp_train_iters=lsbo_cfg.get('gp_train_iters', 100),
            gp_lr=lsbo_cfg.get('gp_lr', 0.1),
            gp_max_train=lsbo_cfg.get('gp_max_train', 512),
            s_star_quantile=lsbo_cfg.get('s_star_quantile', 0.95),
            gp_train_top_frac=lsbo_cfg.get('gp_train_top_frac', 0.5),
            rerank_round_trip=lsbo_cfg.get('rerank_round_trip', True),
        )
        if lsbo_cfg.get('downstream_rerank', True) and new_feats:
            new_feats = _downstream_rerank_lsbo(
                new_feats, env,
                top_n=lsbo_cfg.get('downstream_rerank_top_n', 0),
                cap_rows=lsbo_cfg.get('downstream_rerank_cap_rows', 10000),
            )
        before_count = feature_generator.new_feat
        is_changed = feature_generator.append_new(new_feats)
        accepted = feature_generator.new_feat - before_count
        log(f"[LSBO] append_new: {accepted}/{len(new_feats)} accepted "
            f"({int(is_changed.sum())} unique+valid+novel). "
            f"Rejected = duplicates of pool/new_feats or invalid syntax.")
    else:
        nfo.set_mode(Mode.INFER)
        nfo.eval()
        original_top_feats = feature_generator.get_topk(args.top_feat)
        seqs = tokenizer.transform(original_top_feats)
        epoch = 0
        langevin_tau = float(config.langevin_tau or 0.0)
        if langevin_tau > 0:
            log(f"[Langevin-INFER] tau={langevin_tau} (noise injected per gradient step)")
        else:
            log(f"[INFER] deterministic gradient ascent (langevin_tau=0)")
        before_count = feature_generator.new_feat
        for opt_eta in np.arange(config.feat_opt_eta, config.feat_opt_eta * 10, config.feat_opt_eta * config.feat_opt_eta_inc):
            _, new_seqs = nfo(
                torch.tensor(seqs, dtype=torch.long, device=device),
                opt_eta=opt_eta,
                score_threshold=config.score_threshold,
                langevin_tau=langevin_tau,
            )
            new_feats = tokenizer.inverse_transform(new_seqs)
            is_changed = feature_generator.append_new(new_feats)
            # seqs = seqs[~is_changed]
            if feature_generator.new_feat > args.new_feat:
                break
            epoch += 1
            if epoch > config.num_epoch:
                break
        accepted = feature_generator.new_feat - before_count
        log(f"[INFER] append_new total: {accepted} unique+valid+novel features added")

    if config.use_crossover:
        from autolearn.feat_selection.nfo.crossover import latent_crossover
        cross_cfg = config.crossover or {}
        nfo.eval()
        before_count = feature_generator.new_feat
        before_len = len(feature_generator.new_feats)
        cross_feats = latent_crossover(
            nfo, tokenizer, device,
            pool_features=feature_generator.features,
            pool_scores=feature_generator.scores,
            padding_idx=tokenizer.padding_idx,
            top_k_pairs=cross_cfg.get('top_k_pairs', 8),
            alphas=cross_cfg.get('alphas', [0.25, 0.5, 0.75]),
        )
        feature_generator.append_new(cross_feats)
        accepted = feature_generator.new_feat - before_count
        # tag the newly appended ones as 'crossover' so we can attribute later
        for feat in feature_generator.new_feats[before_len:]:
            feature_generator.meta[feat] = 'crossover'
        log(f"[CROSSOVER] append_new: {accepted}/{len(cross_feats)} accepted")

    feature_generator.pad_new_feats(args.new_feat)

    log(f"{feature_generator.new_feats}", level='debug')
    feature_generator.append(feature_generator.new_feats)
    log(f"Total feature pool size after iter {args.cur_iter}: {len(feature_generator)}")

    num_top_feats = max(args.new_feat, 32)
    log(f"use {num_top_feats} new features with original {len(env.features)} features")

    feature_generator.save()

    if args.score_mode in COMPOSITE_MODES:
        # Downstream eval is post-hoc (run by eval_quality.py). Inside the
        # search loop just rank by composite and report the top usefulness;
        # saves ~len(env.features) RF CV runs per iteration.
        from autolearn.feat_selection.nfo.metrics import composite_from_pool, kext as kext_fn
        n_base = len(env.features)
        n_ops = len(env.ops)
        all_features = feature_generator.features
        all_raw_u = feature_generator.scaler.inverse_transform(
            np.asarray(feature_generator.scores).reshape(-1, 1)
        ).reshape(-1)
        all_trees = generate_tree_from_str(all_features, env)
        all_kext = np.array([
            kext_fn(t, n_base, n_ops) if t is not None else float('nan')
            for t in all_trees
        ], dtype=np.float64)
        all_composite = composite_from_pool(all_raw_u, all_kext, args.lam)
        order = np.argsort(all_composite)[::-1]
        if args.select_k is not None:
            topk = min(args.select_k, len(order))
        else:
            topk = min(max(args.new_feat, len(env.features)), len(order))
        sel = order[:topk]
        max_u = float(all_raw_u[sel].max())
        max_idx = int(np.argmax(all_raw_u[sel])) + 1
        u_label = {'pearson': 'Pearson', 'fisher': 'Fisher',
                   'downstream': 'RF-CV', 'split': 'Otsu',
                   'split_ca': 'ClusterARI', 'anom_auc': 'AnomAUC',
                   'anom_tail': 'TailContrast',
                   'subgroup': 'SubgroupD',
                   'subgroup_op': 'SubgroupD-Op',
                   'rule_out': 'RuleOut',
                   'g3out': 'G3Out',
                   'normout': 'NormOut',
                   'normg3': 'NormG3',
                   'ifout': 'IFOut'}.get(args.score_mode, 'MI')
        log(f"[B-MAFE] iter {args.cur_iter}: top composite = {all_composite[sel[0]]:.4f}, "
            f"top raw {u_label} in top-{topk} = {max_u:.4f}")
        return (max_u, max_idx, args.cur_iter)

    scores = []
    max_score = None
    topk = len(env.features)
    for i in range(topk):
        top_i = i + 1
        features = feature_generator.get_topk(top_i, duplicated=True)
        features = generate_tree_from_str(features, env)
        score = env.eval_set(features)
        scores.append([score])
        if max_score is None or score > max_score[0]:
            max_score = (score, top_i)
    log(f"New max score {max_score}, cur iter: {args.cur_iter}")
    log(f"Original score {env.eval_set([])}")
    return (*max_score, args.cur_iter)


@timeit
def iter_main():
    cur_max = None
    _args = parse_args()
    results_path = None
    if _args.feat_pool:
        os.makedirs(_args.feat_pool, exist_ok=True)
        results_path = os.path.join(_args.feat_pool, 'results.csv')
        if not os.path.exists(results_path):
            with open(results_path, 'w') as f:
                f.write("iter,score,top_k,method\n")
    method_label = _args.method or 'config'
    for i in range(_args.total_iter):
        _args.scratch = _args.cur_iter % 10 == 0
        record = main(_args)
        if cur_max is None or cur_max[0] < record[0]:
            cur_max = record
        if results_path:
            with open(results_path, 'a') as f:
                f.write(f"{record[2]},{record[0]},{record[1]},{method_label}\n")
        _args.cur_iter += 1
    print(cur_max)
    if results_path:
        log(f"per-iter results appended to {results_path}")

    # Cache the final ranked feature list so eval_quality.py can score it
    # against multiple downstream models without re-running the search.
    if _args.score_mode in COMPOSITE_MODES and _args.feat_pool:
        _write_selected_topk(_args)
    elif _args.score_mode == 'cv' and _args.feat_pool:
        _write_cv_manifest(_args)


def _write_cv_manifest(args):
    """cv-mode (original DIFER) counterpart of ``_write_selected_topk``.

    Reads the final pool dump ({total_iter}.json → {feature_str: cv_score}),
    ranks by score, writes an ``eval_manifest.json`` in the legacy VI-cache
    shape ({'features': [...], 'method': ..., 'lam': null}) so eval_quality
    can materialize the columns and run downstream evaluation.
    """
    import json

    pool_path = os.path.join(args.feat_pool, f'{args.cur_iter}.json')
    if not os.path.exists(pool_path):
        log(f"[cv-manifest] no pool file at {pool_path}; skipping eval_manifest.json",
            level='warn')
        return
    with open(pool_path, 'r') as f:
        all_feats = json.load(f)

    items = [(k, float(v)) for k, v in all_feats.items() if k != '']
    items.sort(key=lambda kv: kv[1], reverse=True)
    # Cap at the larger of new_feat and 4x base count (mirrors the mi-mode cap).
    topk = max(args.new_feat, 32)
    items = items[:topk]

    out = {
        'dataset': args.data,
        'method': args.method_label or args.method or 'infer',
        'lam': None,
        'features': [k for k, _ in items],
        'scores': [v for _, v in items],
        'score_mode': 'cv',
    }
    out_path = os.path.join(args.feat_pool, 'eval_manifest.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    log(f"[cv-manifest] wrote top-{len(items)} features to {out_path}")


def _resolve_downstream_model(args, config):
    """Resolve the downstream model for score_mode='downstream':
    CLI --downstream_model wins, else config.downstream_model, else 'RF'."""
    if getattr(args, 'downstream_model', None):
        return args.downstream_model
    cfg = config.downstream_model
    return cfg if cfg in ('RF', 'LIN') else 'RF'


def _greedy_downstream_select(env, feature_strs, raw_u, *, shortlist=64, k=16,
                              n_repeats=3, val_frac=0.3, patience=4, seed=0,
                              selector='forward', n_estimators=None):
    """Greedy forward selection of derived features by held-out downstream RF.

    Plain composite ranking (score_mode='downstream', lam=1) takes the top-k
    features by their *individual* usefulness U(f)=RF-CV([base+f]). That signal
    is noisy (one feature barely moves RF CV) and ignores redundancy — two
    features that each help alone may carry the same information. This grows the
    feature set one at a time, each round adding the candidate that most improves
    an RF trained on [base + already-selected + c], scored on repeated held-out
    splits that are independent of eval_quality's final CV (so the selection does
    not overfit the same folds it is judged on — the failure mode behind the
    German regression).

    Returns the selected indices into ``feature_strs`` in add order (most
    valuable first), or None if no candidate column could be built.

    Cost is bounded to ~shortlist*k*n_repeats RF fits by only considering the
    top-``shortlist`` candidates by individual U.
    """
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, r2_score
    from autolearn.feat_selection.env import make_downstream_model

    task = env.dataset.task
    dm = getattr(env, 'downstream_model', 'RF')
    # RF can't take NaN -> fill the -1 sentinel; the LIN pipeline imputes, so
    # leave NaN/inf in place for it.
    fill_sentinel = (dm != 'LIN')
    y = np.asarray(env.dataset.labels)

    X_base = env.dataset.instances.copy()
    X_base.columns = X_base.columns.astype(str)
    base_np = X_base.to_numpy(dtype=np.float64, copy=True)
    base_np[~np.isfinite(base_np)] = -1.0 if fill_sentinel else np.nan
    n_rows = base_np.shape[0]

    # shortlist by individual usefulness, then build columns (drop unbuildable)
    order = np.argsort(raw_u)[::-1]
    short_idx = [int(i) for i in order[:min(shortlist, len(order))]]
    short_trees = generate_tree_from_str([feature_strs[i] for i in short_idx], env)
    cand_cols, cand_global = [], []
    for j, t in enumerate(short_trees):
        if t is None:
            continue
        try:
            col = np.asarray(env.construct_feature(t), dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if col.shape[0] != n_rows:
            continue
        col[~np.isfinite(col)] = -1.0 if fill_sentinel else np.nan
        cand_cols.append(col)
        cand_global.append(short_idx[j])
    if not cand_cols:
        return None

    # fixed repeated held-out splits (stratified for classification)
    idx_all = np.arange(n_rows)
    splits = []
    for r in range(n_repeats):
        strat = y if task == 'classification' else None
        try:
            tr, va = train_test_split(idx_all, test_size=val_frac,
                                      random_state=seed + r, stratify=strat)
        except ValueError:
            tr, va = train_test_split(idx_all, test_size=val_frac,
                                      random_state=seed + r)
        splits.append((tr, va))

    # Selection RF can use MORE trees than the cheap pool-scoring RF: the greedy
    # evaluates far fewer feature sets than the 4096-feature pool, so a higher
    # n_estimators here buys a much lower-variance selection signal (less
    # overfitting of the held-out criterion) at modest cost.
    sel_n_est = int(n_estimators) if n_estimators else env.n_estimators

    def _mk():
        return make_downstream_model(task, dm, n_estimators=sel_n_est)

    def _score(extra_cols):
        if extra_cols:
            X = np.concatenate([base_np] + [c.reshape(-1, 1) for c in extra_cols], axis=1)
        else:
            X = base_np
        vals = []
        for tr, va in splits:
            m = _mk()
            m.fit(X[tr], y[tr])
            pred = m.predict(X[va])
            if task == 'regression':
                vals.append(r2_score(y[va], pred))
            else:
                vals.append(f1_score(y[va], pred, average='micro'))
        return float(np.mean(vals))

    def _score_ids(ids):
        return _score([cand_cols[i] for i in ids])

    base_score = _score([])
    target = min(k, len(cand_cols))
    selected, remaining = [], set(range(len(cand_cols)))
    best = (base_score, [])             # (score, subset) — best subset by score
    best_size = {0: base_score}         # best score seen at each subset size
    no_improve = 0
    sffs = (selector == 'sffs')
    tag = 'SFFS' if sffs else 'fwd'
    log(f"[greedy/{tag}] base held-out score = {base_score:.4f} "
        f"(shortlist={len(cand_cols)}, k={target}, repeats={n_repeats}, patience={patience})")

    while len(selected) < target and remaining:
        # --- forward: add the best remaining candidate ---
        add_lid, add_val = None, -np.inf
        for lid in remaining:
            v = _score_ids(selected + [lid])
            if v > add_val:
                add_val, add_lid = v, lid
        selected.append(add_lid)
        remaining.discard(add_lid)
        cur = add_val
        sz = len(selected)
        best_size[sz] = max(best_size.get(sz, -np.inf), cur)
        improved = cur > best[0] + 1e-6
        if improved:
            best = (cur, list(selected))
            no_improve = 0
        else:
            no_improve += 1
        log(f"[greedy/{tag}] +{feature_strs[cand_global[add_lid]][:30]:<30} "
            f"size={sz:2d} score={cur:.4f}{'  *best' if improved else ''}")

        # --- SFFS only: conditionally remove redundant features ---
        if sffs:
            while len(selected) > 2:
                rm_lid, rm_val = None, -np.inf
                for f in selected:
                    v = _score_ids([x for x in selected if x != f])
                    if v > rm_val:
                        rm_val, rm_lid = v, f
                if rm_val > best_size.get(len(selected) - 1, -np.inf) + 1e-6:
                    selected.remove(rm_lid)
                    remaining.add(rm_lid)
                    cur = rm_val
                    sz = len(selected)
                    best_size[sz] = max(best_size.get(sz, -np.inf), cur)
                    if cur > best[0] + 1e-6:
                        best = (cur, list(selected))
                        no_improve = 0
                    log(f"[greedy/{tag}] -float -> size={sz:2d} score={cur:.4f}")
                else:
                    break

        if no_improve >= patience:
            log(f"[greedy/{tag}] early stop: {patience} non-improving rounds")
            break

    # Base-only fallback: if NO subset beat the base held-out score, keep zero
    # derived features (best[1] == []). The caller then writes a 0-feature
    # manifest and the downstream eval uses base features alone — guaranteeing
    # B-MAFE >= base on saturated datasets instead of forcing in a feature that
    # hurts. (Returns [] for base-only vs None only on construction failure.)
    keep_ids = list(best[1])
    log(f"[greedy/{tag}] keeping {len(keep_ids)} feature(s)"
        f"{' (base-only: nothing beat base)' if not keep_ids else ''}, "
        f"held-out {best[0]:.4f} (base {base_score:.4f}, gain {best[0]-base_score:+.4f})")
    return [cand_global[i] for i in keep_ids]


def _write_selected_topk(args):
    """Read the latest pool JSON and emit selected_topk.json ranked by composite."""
    import json
    from autolearn.feat_selection.nfo.metrics import composite_from_pool, kext as kext_fn

    set_cuda(args.cuda)
    file_path = os.path.abspath(os.path.dirname(__file__))
    config = get_config(f"{file_path}/config/{args.hyper_config}.json")
    force_num_ops = bool(getattr(args, 'num_ops_only', False)) or bool(
        config.num_ops_only if config.num_ops_only is not None else False
    )
    env = SklearnEnv(
        args.data, args.max_order if args.max_order is not None else config.max_order,
        model=args.eval_model,
        cv=args.cv, parallel=args.parallel_eval,
        n_estimators=args.n_estimators, scoring=args.c_scoring,
        score_mode=args.score_mode,
        downstream_model=_resolve_downstream_model(args, config),
        use_cmi=bool(config.use_cmi) if config.use_cmi is not None else False,
        use_cat_space=bool(config.use_cat_space) if config.use_cat_space is not None else True,
        force_num_ops=force_num_ops,
        ops_subset=args.ops.split(',') if args.ops else None,
        subgroup_rule=getattr(args, 'subgroup_rule', 0),
    )

    # The latest pool was saved by FeatureGenerator.save() at the end of the
    # final iteration. cur_iter was already incremented past that, so the file
    # we want is `{feat_pool}/{cur_iter}.json` (= total_iter when starting at 0).
    pool_path = os.path.join(args.feat_pool, f'{args.cur_iter}.json')
    if not os.path.exists(pool_path):
        log(f"[B-MAFE] no pool file at {pool_path}; skipping selected_topk.json", level='warn')
        return
    with open(pool_path, 'r') as f:
        all_feats = json.load(f)

    feature_strs = [k for k in all_feats.keys() if k != '']
    raw_u = np.array([all_feats[k] for k in feature_strs], dtype=np.float64)
    n_base = len(env.features)
    n_ops = len(env.ops)
    trees = generate_tree_from_str(feature_strs, env)
    kext_scores = np.array([
        kext_fn(t, n_base, n_ops) if t is not None else float('nan')
        for t in trees
    ], dtype=np.float64)
    composite = composite_from_pool(raw_u, kext_scores, args.lam)
    order = np.argsort(composite)[::-1]
    if getattr(args, 'select_k', None) is not None:
        topk = min(args.select_k, len(order))
    else:
        topk = min(max(args.new_feat, len(env.features) * 4), len(order))
    sel = [int(i) for i in order[:topk]]

    # Set-level objective (downstream mode only): replace the composite top-k
    # — which ranks features by their noisy *individual* marginal CV — with a
    # greedy forward selection that grows the set by held-out downstream RF gain,
    # accounting for redundancy and avoiding same-fold selection overfit.
    selection_method = 'composite_topk'
    greedy_cfg = config.downstream_greedy or {}
    if args.score_mode == 'downstream' and greedy_cfg.get('enable', True):
        selector = (getattr(args, 'greedy_selector', None)
                    or greedy_cfg.get('selector', 'forward'))
        gsel = _greedy_downstream_select(
            env, feature_strs, raw_u,
            shortlist=int(greedy_cfg.get('shortlist', 64)),
            k=int(greedy_cfg.get('k', 16)),
            n_repeats=int(greedy_cfg.get('n_repeats', 3)),
            val_frac=float(greedy_cfg.get('val_frac', 0.3)),
            patience=int(greedy_cfg.get('patience', 4)),
            selector=selector,
            n_estimators=greedy_cfg.get('n_estimators'),
        )
        # gsel is None only on construction failure (-> keep composite top-k);
        # an empty list is a valid base-only selection (nothing beat base).
        if gsel is not None:
            sel = gsel
            topk = len(sel)
            selection_method = 'greedy_setlevel'
            log(f"[B-MAFE] downstream greedy set-level selection: {topk} features"
                + (" (base-only)" if topk == 0 else ""))
        else:
            log("[B-MAFE] greedy set-level selection failed; "
                "falling back to composite top-k", level='warn')

    _USEFULNESS_FIELD = {'pearson': 'pearson', 'fisher': 'fisher',
                         'downstream': 'downstream_rf_cv'}.get(
        args.score_mode, 'mi')
    out = {
        'dataset': args.data,
        'lam': args.lam,
        'method': args.method_label or args.method or 'vi',
        'score_mode': args.score_mode,
        'usefulness_metric': _USEFULNESS_FIELD,
        'selection_method': selection_method,
        'top_k': topk,
        'features': [feature_strs[i] for i in sel],
        # Field name kept as 'raw_mi' for backward compatibility with the
        # existing eval_quality reader; under score_mode in {'pearson','fisher'}
        # the values are the corresponding usefulness metric, disambiguated by
        # 'usefulness_metric' above.
        'raw_mi': [float(raw_u[i]) for i in sel],
        'kext': [float(kext_scores[i]) for i in sel],
        'composite': [float(composite[i]) for i in sel],
    }
    out_path = os.path.join(args.feat_pool, f'selected_topk_lam{args.lam}.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    log(f"[B-MAFE] wrote selected top-{topk} features (lam={args.lam}) to {out_path}")


if __name__ == '__main__':
    iter_main()

