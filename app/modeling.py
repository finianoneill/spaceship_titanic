"""
Cross-validation, model factories and submission helpers.

Every model is wrapped as ``fit_predict(X_train, y_train, X_valid) -> proba`` so
the same CV loop, out-of-fold predictions and ensembling work for all of them.
"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
N_SPLITS = 5


def cv_splits(y, groups, n_splits=N_SPLITS, seed=SEED):
    """Fixed StratifiedGroupKFold splits (groupmates never straddle a split)."""
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y, groups))


def cross_validate(fit_predict, X, y, splits, name='model', threshold=0.5, verbose=True):
    """Run CV. Returns dict with fold accuracies, mean/std and OOF probabilities."""
    oof = np.zeros(len(y))
    scores = []
    for tr_idx, va_idx in splits:
        proba = fit_predict(X.iloc[tr_idx], y.iloc[tr_idx], X.iloc[va_idx])
        oof[va_idx] = proba
        scores.append(accuracy_score(y.iloc[va_idx], proba >= threshold))
    res = {'name': name, 'cv_mean': float(np.mean(scores)), 'cv_std': float(np.std(scores)),
           'folds': scores, 'oof': oof, 'features': list(X.columns)}
    if verbose:
        print(f'{name:<40} CV accuracy {res["cv_mean"]:.4f} ± {res["cv_std"]:.4f}  '
              f'folds {np.round(scores, 4)}')
    return res


def results_table(results):
    """Summary table from a list of cross_validate results."""
    return (pd.DataFrame([{'Experiment': r['name'], 'CV mean': r['cv_mean'], 'CV std': r['cv_std'],
                           'N features': len(r['features'])} for r in results])
              .set_index('Experiment').round(4))


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------
def logreg_fit_predict(categorical, C=1.0):
    """One-hot categoricals + scaled numerics -> logistic regression."""
    def fit_predict(X_tr, y_tr, X_va):
        cats = [c for c in categorical if c in X_tr.columns]
        nums = [c for c in X_tr.columns if c not in cats]
        pre = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), cats),
            ('num', StandardScaler(), nums),
        ])
        model = make_pipeline(pre, LogisticRegression(C=C, max_iter=2000))
        # numpy 2.x on Apple Accelerate BLAS emits spurious matmul overflow
        # RuntimeWarnings here (inputs are scaled, |x| < 7; coefficients finite).
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            model.fit(X_tr, y_tr)
            return model.predict_proba(X_va)[:, 1]
    return fit_predict


def catboost_fit_predict(categorical, **params):
    """CatBoost with native categorical handling."""
    from catboost import CatBoostClassifier
    base = dict(iterations=1000, learning_rate=0.03, depth=6, random_seed=SEED,
                verbose=0, allow_writing_files=False, thread_count=-1)
    base.update(params)

    def fit_predict(X_tr, y_tr, X_va):
        cats = [c for c in categorical if c in X_tr.columns]
        model = CatBoostClassifier(**base, cat_features=cats)
        model.fit(X_tr, y_tr)
        fit_predict.last_model = model
        return model.predict_proba(X_va)[:, 1]
    return fit_predict


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------
def write_submission(test_ids, proba, name, cv_score, threshold=0.5, out_dir='submissions'):
    """Write a Kaggle submission CSV named ``<name>_cv<score>.csv``."""
    os.makedirs(out_dir, exist_ok=True)
    sub = pd.DataFrame({'PassengerId': test_ids, 'Transported': proba >= threshold})
    path = os.path.join(out_dir, f'{name}_cv{cv_score:.4f}.csv')
    sub.to_csv(path, index=False)
    print(f'Wrote {path}  ({len(sub)} rows, {sub["Transported"].mean():.1%} predicted transported)')
    return path
