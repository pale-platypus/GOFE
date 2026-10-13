"""Create frozen train/test dataset splits for the FAIR comparison.

For each dataset <name>, write:
  data/<name>__tr.csv / .json   — TRAIN rows (all feature search + selection
                                  for every method runs on this dataset only)
  data/<name>__te.csv / .json   — TEST rows (held out; only the final eval
                                  touches it)

Same CSV format the loader expects (header=None, last column = label) and the
same meta JSON (column types + task), so every existing run script works
unchanged with --data=<name>__tr.

Split: 70/30, fixed seed, stratified on the label for classification. This is
the single most important fairness fix: no method's feature generation or
selection ever sees the test rows.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

DATA = 'data'
DATASETS = ['pima', 'german', 'housing', '620']
SEED = 42
TEST_FRAC = 0.30


def split_one(name):
    csv = f'{DATA}/{name}.csv'
    meta = f'{DATA}/{name}.json'
    if not (os.path.exists(csv) and os.path.exists(meta)):
        return f'skip ({name}: missing csv/json)'
    df = pd.read_csv(csv, header=None)
    with open(meta) as f:
        m = json.load(f)
    task = m.get('task')
    y = df.iloc[:, -1]
    strat = y if task == 'C' else None
    tr, te = train_test_split(df, test_size=TEST_FRAC, random_state=SEED,
                              stratify=strat)
    tr.to_csv(f'{DATA}/{name}__tr.csv', header=False, index=False)
    te.to_csv(f'{DATA}/{name}__te.csv', header=False, index=False)
    shutil.copy(meta, f'{DATA}/{name}__tr.json')
    shutil.copy(meta, f'{DATA}/{name}__te.json')
    return f'OK ({name}: train={len(tr)} test={len(te)} task={task})'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=DATASETS)
    args = p.parse_args()
    for d in args.datasets:
        print(split_one(d))


if __name__ == '__main__':
    main()
