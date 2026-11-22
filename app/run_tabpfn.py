"""
Compute TabPFN out-of-fold and test probabilities in a standalone process.

Why a separate process: on macOS, PyTorch (TabPFN) and CatBoost each ship their
own OpenMP runtime; once CatBoost has run, initialising TabPFN in the same
process segfaults. Running TabPFN here and caching its predictions avoids that,
and also avoids recomputing ~10 minutes of inference on every notebook run.

Uses exactly the same preprocessing and the same StratifiedGroupKFold seeds as
the notebook (``mdl.EVAL_SEEDS``), so results are directly comparable/blendable.

Usage (from the repo root):
    python -m app.run_tabpfn --features full
    python -m app.run_tabpfn --features compact --version v3.5
    python -m app.run_tabpfn --finetune --seeds 0 --no-test   # fine-tuned pilot

Output: artifacts/tabpfn_<version>_<features>.npz with
    oofs (n_seeds x n_train), seeds, per_seed, test_proba, features
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from app import modeling as mdl
from app import preprocessing as pp

# Compact set: raw columns + a few EDA features, for a foundation model that does
# its own preprocessing and may prefer less redundancy.
COMPACT_FEATURES = [
    'HomePlanet', 'Destination', 'Deck', 'Side',
    'CryoSleep_int', 'VIP_int', 'Age', 'Cabin_num', 'GroupSize',
    'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'LuxuryShare',
]
FEATURE_SETS = {'full': pp.FEATURES, 'compact': COMPACT_FEATURES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', choices=FEATURE_SETS, default='full')
    ap.add_argument('--version', default='v3.5')
    ap.add_argument('--spend-fill', default='zero')
    ap.add_argument('--seeds', type=int, nargs='+', default=mdl.EVAL_SEEDS)
    ap.add_argument('--out-dir', default='artifacts')
    ap.add_argument('--finetune', action='store_true', help='fine-tuned TabPFN (~4 min/fold on M4)')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--no-test', action='store_true', help='skip the full-train test fit')
    args = ap.parse_args()
    warnings.filterwarnings('ignore')

    feats = FEATURE_SETS[args.features]
    cats = [c for c in pp.CATEGORICAL_FEATURES if c in feats]
    full, _ = pp.preprocess(pd.read_csv('data/train.csv'), pd.read_csv('data/test.csv'),
                            spend_fill=args.spend_fill, verbose=False)
    X, y, groups, X_test, _ = pp.split_xy(full, feats)

    oofs, per_seed, test_probas = [], [], []
    for s in args.seeds:
        t = time.time()
        if args.finetune:
            fp = mdl.tabpfn_finetuned_fit_predict(cats, version=args.version, random_state=s,
                                                  epochs=args.epochs)
        else:
            fp = mdl.tabpfn_fit_predict(cats, version=args.version, random_state=s)
        res = mdl.cross_validate(fp, X, y, mdl.cv_splits(y, groups, seed=s), verbose=False)
        oofs.append(res['oof'])
        per_seed.append(accuracy_score(y, res['oof'] >= 0.5))
        if not args.no_test:
            test_probas.append(fp(X, y, X_test))  # full-train fit for the submission
        print(f'seed {s}: OOF acc {per_seed[-1]:.4f}  ({time.time() - t:.0f}s)', flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = '_ft' if args.finetune else ''
    seeds_tag = '' if args.seeds == mdl.EVAL_SEEDS else '_s' + '-'.join(map(str, args.seeds))
    path = os.path.join(args.out_dir, f'tabpfn_{args.version}{tag}_{args.features}{seeds_tag}.npz')
    np.savez(path, oofs=np.array(oofs), seeds=np.array(args.seeds), per_seed=np.array(per_seed),
             test_proba=np.mean(test_probas, axis=0) if test_probas else np.array([]),
             features=np.array(feats))
    print(f'TabPFN {args.version}{tag} [{args.features}]: OOF acc {np.mean(per_seed):.4f} '
          f'± {np.std(per_seed):.4f} -> {path}')


if __name__ == '__main__':
    main()
