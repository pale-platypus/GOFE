"""Shared helpers for the DATA-UNDERSTANDING experiments (Exp 1/2/3).

Everything operates on the train split (<name>__tr) so feature discovery never
sees test. Matching of discovered features to canonical constructs is by
VALUE-based monotone equivalence (|Spearman| ~ 1), which robustly captures
"monotone-transform equivalents count" without fragile symbolic simplification;
readable infix is for display only.
"""
from __future__ import annotations
import glob
import json
import os
import re
import sys
import numpy as np

sys.path.insert(0, '.')
from autolearn.feat_selection.env import SklearnEnv
from autolearn.feat_selection.nfo.feat_tree import generate_tree_from_str
from autolearn.feat_selection.nfo.metrics import kext as kext_fn, mutual_information
from scipy.stats import spearmanr

# Pre-registered canonical constructs (VERIFIED against column ranges, Step 0).
# postfix = B-MAFE grammar; names map 0-indexed base cols to domain names.
CONSTRUCTS = {
    'pima':            dict(postfix='1,4,*',        readable='Glucose * Insulin',
                            names={1: 'Glucose', 4: 'Insulin'}),
    'housing':         dict(postfix='5,5,*',        readable='RM^2',
                            names={5: 'RM'}),
    'german':          dict(postfix='4,1,division', readable='amount / duration',
                            names={4: 'amount', 1: 'duration'}),
    'winequality-red': dict(postfix='5,6,division', readable='free_SO2 / total_SO2',
                            names={5: 'free_SO2', 6: 'total_SO2'}),
    'airfoil':         dict(postfix='0,4,*,3,division', readable='freq * disp_thick / velocity',
                            names={0: 'frequency', 3: 'velocity', 4: 'disp_thickness'}),
}

_ENV = {}
_OP_SYMBOL = {'+': '+', '-': '-', '*': '*', 'division': '/', 'reciprocal': '1/',
              'log': 'log', 'sqrt_abs': 'sqrt|.|', 'min_max': 'minmax', 'mod_column': 'mod'}


def get_env(name, max_order=4):
    if name not in _ENV:
        _ENV[name] = SklearnEnv(name, max_order, parallel=False,
                                score_mode='mi', force_num_ops=True)
    return _ENV[name]


def n_base_ops(env):
    return len(env.features), len(env.ops)


def construct_values(env, postfix):
    """Evaluate a postfix expression on env's data -> np array (or None)."""
    trees = generate_tree_from_str([postfix], env)
    t = trees[0] if trees else None
    if t is None:
        return None
    try:
        col = np.asarray(env.construct_feature(t), dtype=np.float64).reshape(-1)
    except Exception:
        return None
    return col


def feat_K(env, postfix):
    trees = generate_tree_from_str([postfix], env)
    t = trees[0] if trees else None
    if t is None:
        return float('nan')
    nb, no = n_base_ops(env)
    return float(kext_fn(t, nb, no))


def feat_MI(env, values):
    if values is None:
        return float('nan')
    v = np.where(np.isfinite(values), values, np.nan)
    if np.all(np.isnan(v)):
        return float('nan')
    med = np.nanmedian(v)
    v = np.where(np.isnan(v), med, v)
    y = np.asarray(env.dataset.labels)
    return float(mutual_information(v, y, env.dataset.task))


def monotone_match(values_a, values_b):
    """|Spearman| between two feature vectors (nan-safe). 1.0 = monotone-equiv."""
    if values_a is None or values_b is None:
        return 0.0
    a = np.asarray(values_a, float); b = np.asarray(values_b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return 0.0
    r = spearmanr(a[m], b[m]).correlation
    return abs(r) if np.isfinite(r) else 0.0


def load_pool(tree_root, dataset_tr, tag='mi_gat', lam='0.5'):
    """Full evaluated pool {postfix: raw_mi} from the latest {N}.json."""
    d = f'{tree_root}/nfo/RF/lsbo/{tag}/lam{lam}/{dataset_tr}'
    js = [p for p in glob.glob(os.path.join(d, '*.json'))
          if re.fullmatch(r'\d+', os.path.splitext(os.path.basename(p))[0])]
    if not js:
        return {}, d
    latest = max(js, key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    with open(latest) as f:
        pool = json.load(f)
    return {k: v for k, v in pool.items() if k != ''}, latest


def to_infix(postfix, names=None):
    """Readable infix from a postfix token string (display only)."""
    st = []
    names = names or {}
    for tok in postfix.split(','):
        if tok in _OP_SYMBOL:
            sym = _OP_SYMBOL[tok]
            if tok in ('log', 'sqrt_abs', 'reciprocal', 'min_max'):
                a = st.pop() if st else '?'
                st.append(f'{sym}({a})' if tok != 'reciprocal' else f'1/({a})')
            else:
                b = st.pop() if st else '?'; a = st.pop() if st else '?'
                st.append(f'({a} {sym} {b})')
        else:
            st.append(names.get(int(tok), f'f{tok}') if tok.isdigit() else tok)
    return st[-1] if st else postfix


def to_mathtext(postfix, names=None):
    """Matplotlib mathtext (no surrounding $) from a postfix token string.

    Renders sqrt_abs as a true radical (\\sqrt{|.|}); keeps division/reciprocal
    inline (a/b) rather than tall \\frac so annotations stay single-height.
    Parenthesization is precedence-aware: parens are emitted only where they
    change meaning (function delimiters and | .| bars already group, so no
    redundant parens inside them or around the whole expression).
    """
    ATOM, MUL, ADD = 4, 2, 1     # operator precedence levels
    st = []                       # stack of (text, precedence)
    names = names or {}

    def esc(s):
        return str(s).replace('_', r'\_').replace('%', r'\%')

    def wrap(child, parent, right=False, nonassoc=False):
        t, p = child
        if p < parent or (nonassoc and right and p == parent):
            return '(' + t + ')'
        return t

    for tok in postfix.split(','):
        if tok in _OP_SYMBOL:
            if tok in ('log', 'sqrt_abs', 'reciprocal', 'min_max'):
                a = st.pop() if st else ('?', ATOM)
                at = a[0]
                if tok == 'sqrt_abs':
                    st.append((r'\sqrt{|' + at + r'|}', ATOM))
                elif tok == 'log':
                    st.append((r'\log(' + at + ')', ATOM))
                elif tok == 'min_max':
                    st.append((r'\mathrm{minmax}(' + at + ')', ATOM))
                else:  # reciprocal -> 1/a
                    st.append(('1/' + wrap(a, MUL, right=True, nonassoc=True), MUL))
            else:
                b = st.pop() if st else ('?', ATOM)
                a = st.pop() if st else ('?', ATOM)
                if tok == '*':
                    st.append((wrap(a, MUL) + r' \cdot ' + wrap(b, MUL, right=True), MUL))
                elif tok == 'division':
                    st.append((wrap(a, MUL, nonassoc=True) + '/'
                               + wrap(b, MUL, right=True, nonassoc=True), MUL))
                elif tok == 'mod_column':
                    st.append((wrap(a, MUL, nonassoc=True) + r'\,\mathrm{mod}\,'
                               + wrap(b, MUL, right=True, nonassoc=True), MUL))
                else:  # + or -
                    st.append((wrap(a, ADD) + f' {tok} '
                               + wrap(b, ADD, right=True, nonassoc=(tok == '-')), ADD))
        elif tok.isdigit():
            st.append((r'\mathrm{' + esc(names.get(int(tok), f'f{tok}')) + '}', ATOM))
        else:
            st.append((esc(tok), ATOM))
    return st[-1][0] if st else esc(postfix)


def _vis_len(seg):
    """Approximate on-screen character width of a mathtext fragment."""
    import re
    t = (seg.replace(r'\cdot', '.').replace(r'\sqrt', 'v').replace(r'\log', 'log')
            .replace(r'\mathrm', '').replace(r'\,', '').replace('\\ ', ' '))
    t = re.sub(r'\\[a-zA-Z]+', '', t)
    return len(t.replace('{', '').replace('}', ''))


def wrap_mathtext(mtxt, width=30):
    """Wrap a long mathtext expression across lines for rendering.

    Breaks only at top-level ``+``/``-``/``\\cdot`` operators (brace depth 0, so
    ``\\sqrt{...}`` and ``\\mathrm{...}`` are never split), packs segments
    greedily up to ``width`` visible chars, and returns a ``\\n``-joined string
    with each line wrapped in ``$...$``. Short expressions return a single line.
    """
    ops = (' + ', ' - ', r' \cdot ')
    segs, depth, i, last = [], 0, 0, 0
    while i < len(mtxt):
        c = mtxt[i]
        if c == '{':
            depth += 1; i += 1; continue
        if c == '}':
            depth -= 1; i += 1; continue
        if depth == 0:
            hit = next((op for op in ops if mtxt.startswith(op, i)), None)
            if hit:
                end = i + len(hit)
                segs.append(mtxt[last:end]); last = end; i = end; continue
        i += 1
    segs.append(mtxt[last:])
    segs = [s for s in segs if s]

    lines, cur = [], ''
    for seg in segs:
        if cur and _vis_len(cur) + _vis_len(seg) > width:
            lines.append(cur); cur = seg
        else:
            cur += seg
    if cur:
        lines.append(cur)
    return '\n'.join('$' + ln.strip() + '$' for ln in lines)
