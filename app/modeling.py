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


# ---------------------------------------------------------------------------
# Round 2: multi-seed CV, more models, fold-aware features, ensembling
# ---------------------------------------------------------------------------
EVAL_SEEDS = [0, 1, 2, 3, 4]
TUNING_SEED = 100  # Optuna tunes on these splits; never used for evaluation


def multi_seed_cv(fit_predict, X, y, groups, name, seeds=EVAL_SEEDS, verbose=True):
    """Repeat grouped CV over several split seeds.

    The headline metric is OOF accuracy (all train rows), averaged over seeds.
    Using the same seeds for every experiment makes comparisons *paired*.
    """
    per_seed, oofs = [], []
    for s in seeds:
        res = cross_validate(fit_predict, X, y, cv_splits(y, groups, seed=s), name=name, verbose=False)
        oofs.append(res['oof'])
        per_seed.append(accuracy_score(y, res['oof'] >= 0.5))
    out = {'name': name, 'seeds': list(seeds), 'per_seed': np.array(per_seed), 'oofs': oofs,
           'mean': float(np.mean(per_seed)), 'std': float(np.std(per_seed)), 'features': list(X.columns)}
    if verbose:
        print(f'{name:<45} OOF acc {out["mean"]:.4f} ± {out["std"]:.4f} over {len(seeds)} seeds  '
              f'{np.round(per_seed, 4)}')
    return out


def compare(result, baseline):
    """Paired comparison vs a baseline run on the same seeds."""
    d = result['per_seed'] - baseline['per_seed']
    return {'Experiment': result['name'], 'OOF acc': result['mean'], 'Δ vs baseline': d.mean(),
            'Seeds better': f'{int((d > 0).sum())}/{len(d)}'}


def comparison_table(results, baseline):
    return pd.DataFrame([compare(r, baseline) for r in results]).set_index('Experiment').round(4)


def ensemble_result(results, name, weights=None):
    """Average OOF probabilities of multi-seed results (seed by seed)."""
    weights = np.ones(len(results)) if weights is None else np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    seeds = results[0]['seeds']
    assert all(r['seeds'] == seeds for r in results), 'results must share seeds'
    oofs = [sum(w * r['oofs'][i] for w, r in zip(weights, results)) for i in range(len(seeds))]
    return {'name': name, 'seeds': seeds, 'oofs': oofs, 'features': results[0]['features'],
            'per_seed': None, 'mean': None, 'std': None}


def score_result(result, y, verbose=True):
    """Fill per-seed accuracy for a result built from OOF probabilities."""
    result['per_seed'] = np.array([accuracy_score(y, o >= 0.5) for o in result['oofs']])
    result['mean'], result['std'] = float(result['per_seed'].mean()), float(result['per_seed'].std())
    if verbose:
        print(f'{result["name"]:<45} OOF acc {result["mean"]:.4f} ± {result["std"]:.4f}  '
              f'{np.round(result["per_seed"], 4)}')
    return result


def _to_category(X_tr, X_va, categorical):
    """Consistent pandas categories for LightGBM/XGBoost (no target used)."""
    X_tr, X_va = X_tr.copy(), X_va.copy()
    for c in [c for c in categorical if c in X_tr.columns]:
        dtype = pd.CategoricalDtype(sorted(set(X_tr[c].dropna()) | set(X_va[c].dropna())))
        X_tr[c], X_va[c] = X_tr[c].astype(dtype), X_va[c].astype(dtype)
    return X_tr, X_va


def lgbm_fit_predict(categorical, **params):
    from lightgbm import LGBMClassifier
    base = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=SEED, verbose=-1)
    base.update(params)

    def fit_predict(X_tr, y_tr, X_va):
        X_tr, X_va = _to_category(X_tr, X_va, categorical)
        model = LGBMClassifier(**base).fit(X_tr, y_tr)
        return model.predict_proba(X_va)[:, 1]
    return fit_predict


def xgb_fit_predict(categorical, **params):
    from xgboost import XGBClassifier
    base = dict(n_estimators=500, learning_rate=0.03, max_depth=5, subsample=0.8,
                colsample_bytree=0.8, reg_lambda=1.0, tree_method='hist', enable_categorical=True,
                max_cat_to_onehot=8, random_state=SEED, n_jobs=-1)
    base.update(params)

    def fit_predict(X_tr, y_tr, X_va):
        X_tr, X_va = _to_category(X_tr, X_va, categorical)
        model = XGBClassifier(**base).fit(X_tr, y_tr)
        return model.predict_proba(X_va)[:, 1]
    return fit_predict


# --- Cabin neighbour rate (fold-aware target feature) ----------------------
def neighbor_rate(query, ref, y_ref, k=10, prior=5.0, exclude_own_group=False):
    """Smoothed transported rate of ``ref`` passengers within ±k cabins on the
    same deck and side as each ``query`` row.

    ``exclude_own_group=True`` (use when query is ref) removes each row's own
    group, including itself, so no row sees its own label. Shrinks toward the
    ref mean with strength ``prior``.
    """
    y_ref = np.asarray(y_ref, dtype=float)
    p = y_ref.mean()
    ref = ref[['Deck', 'Side', 'Cabin_num', 'GroupId']].reset_index(drop=True).assign(y=y_ref)
    q = query[['Deck', 'Side', 'Cabin_num', 'GroupId']].reset_index(drop=True)
    s = np.zeros(len(q))
    n = np.zeros(len(q))
    ref_groups = ref.groupby(['Deck', 'Side'])
    for key, qidx in q.groupby(['Deck', 'Side']).indices.items():
        if key not in ref_groups.groups:
            continue
        r = ref_groups.get_group(key).sort_values('Cabin_num')
        nums = r['Cabin_num'].to_numpy()
        cs = np.concatenate([[0.0], np.cumsum(r['y'].to_numpy())])
        qn = q['Cabin_num'].to_numpy()[qidx]
        lo = np.searchsorted(nums, qn - k, side='left')
        hi = np.searchsorted(nums, qn + k, side='right')
        s[qidx] = cs[hi] - cs[lo]
        n[qidx] = hi - lo
    if exclude_own_group:
        pairs = q.reset_index().merge(ref, on='GroupId', suffixes=('', '_r'))
        pairs = pairs[(pairs['Deck'] == pairs['Deck_r']) & (pairs['Side'] == pairs['Side_r'])
                      & ((pairs['Cabin_num'] - pairs['Cabin_num_r']).abs() <= k)]
        own = pairs.groupby('index')['y'].agg(['sum', 'count'])
        s[own.index] -= own['sum'].to_numpy()
        n[own.index] -= own['count'].to_numpy()
    return (s + prior * p) / (n + prior)


def with_neighbor_feature(base_fit_predict, k=10, prior=5.0):
    """Wrap a model: add ``NeighborRate`` using only training-fold labels.

    X must contain Deck, Side, Cabin_num and GroupId; GroupId is dropped before
    the model sees the data.
    """
    def fit_predict(X_tr, y_tr, X_va):
        X_tr, X_va = X_tr.copy(), X_va.copy()
        X_tr['NeighborRate'] = neighbor_rate(X_tr, X_tr, y_tr, k, prior, exclude_own_group=True)
        X_va['NeighborRate'] = neighbor_rate(X_va, X_tr, y_tr, k, prior, exclude_own_group=False)
        proba = base_fit_predict(X_tr.drop(columns='GroupId'), y_tr, X_va.drop(columns='GroupId'))
        fit_predict.last_model = getattr(base_fit_predict, 'last_model', None)
        return proba
    return fit_predict


# --- Optuna ----------------------------------------------------------------
def tune_catboost(X, y, groups, categorical, n_trials=40, wrap=None, seed=TUNING_SEED):
    """Optuna search for CatBoost on dedicated tuning splits (not EVAL_SEEDS)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    splits = cv_splits(y, groups, seed=seed)

    def objective(trial):
        params = dict(
            depth=trial.suggest_int('depth', 4, 8),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            iterations=trial.suggest_int('iterations', 400, 1500, step=100),
            l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1.0, 15.0, log=True),
            random_strength=trial.suggest_float('random_strength', 0.0, 2.0),
            bagging_temperature=trial.suggest_float('bagging_temperature', 0.0, 1.0),
        )
        fp = catboost_fit_predict(categorical, **params)
        fp = wrap(fp) if wrap else fp
        res = cross_validate(fp, X, y, splits, verbose=False)
        return accuracy_score(y, res['oof'] >= 0.5)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    return study


# --- Final fit ---------------------------------------------------------------
def fit_full_predict(fit_predict_factories, X, y, X_test, seeds=EVAL_SEEDS):
    """Fit each model factory on all of train for several random seeds and
    average the test probabilities. Factories take ``seed`` and return a
    fit_predict."""
    probas = []
    for factory in fit_predict_factories:
        probas.append(np.mean([factory(s)(X, y, X_test) for s in seeds], axis=0))
    return np.mean(probas, axis=0)
