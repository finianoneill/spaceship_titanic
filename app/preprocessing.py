"""
Preprocessing for modeling: imputation rules from the EDA + final feature set.

Pipeline (see ``preprocess``):
    combine train/test -> parse raw columns -> deterministic imputation rules
    -> statistical fill (train-row statistics only) -> derived features

The deterministic rules never look at the target, and the fallback statistics
are computed from train rows only, so nothing leaks from test or from labels.
Each rule logs how many values it filled.
"""
import numpy as np
import pandas as pd

from app.eda_helpers import (
    SPEND_COLS, TARGET, add_derived_features, combine_train_test, parse_raw_columns,
)

# Deck -> HomePlanet where the deck is single-planet (EDA section 12)
DECK_TO_PLANET = {'A': 'Europa', 'B': 'Europa', 'C': 'Europa', 'T': 'Europa', 'G': 'Earth'}

CATEGORICAL_FEATURES = [
    'HomePlanet', 'Destination', 'Deck', 'Side', 'Deck_Side', 'Route', 'CryoSleep_Deck',
]
NUMERIC_FEATURES = [
    'CryoSleep_int', 'VIP_int', 'Age', 'Cabin_num', 'GroupSize', 'IsSolo',
    'log_RoomService', 'log_FoodCourt', 'log_ShoppingMall', 'log_Spa', 'log_VRDeck',
    'log_TotalSpending', 'log_LuxurySpending', 'log_BasicSpending',
    'LuxuryShare', 'HasSpent', 'NumAmenitiesUsed',
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def _log(log, rule, n):
    log.append({'Rule': rule, 'Filled': int(n)})


def _fill_from_group(df, col, key, rule, log):
    """Fill ``col`` with the most common non-null value among rows sharing ``key``."""
    mode = (df.dropna(subset=[col, key]).groupby(key)[col]
              .agg(lambda s: s.mode().iloc[0]))
    mask = df[col].isnull() & df[key].isin(mode.index)
    df.loc[mask, col] = df.loc[mask, key].map(mode)
    _log(log, rule, mask.sum())


def apply_imputation_rules(df, log=None):
    """Deterministic imputation rules from the EDA, applied in order.

    Expects the output of ``parse_raw_columns``. Returns (df, log).
    """
    df = df.copy()
    log = [] if log is None else log
    spend_known = df[SPEND_COLS].fillna(0).sum(axis=1)

    # 1. CryoSleep from spending: any spend -> awake; zero spend & adult -> asleep (96%)
    miss = df['CryoSleep'].isnull()
    m = miss & (spend_known > 0)
    df.loc[m, 'CryoSleep'] = False
    _log(log, 'CryoSleep: spend > 0 -> False', m.sum())
    m = miss & (spend_known == 0) & (df['Age'] >= 13)
    df.loc[m, 'CryoSleep'] = True
    _log(log, 'CryoSleep: zero spend & age >= 13 -> True', m.sum())

    # 2. Sleepers and children never spend
    for rule, mask in [('Spend: CryoSleep -> 0', df['CryoSleep'] == True),  # noqa: E712
                       ('Spend: age < 13 -> 0', df['Age'] < 13)]:
        n = 0
        for col in SPEND_COLS:
            m = mask & df[col].isnull()
            df.loc[m, col] = 0.0
            n += m.sum()
        _log(log, rule, n)

    # 3. HomePlanet: groupmates -> family -> single-planet decks
    _fill_from_group(df, 'HomePlanet', 'GroupId', 'HomePlanet: from group', log)
    _fill_from_group(df, 'HomePlanet', 'LastName', 'HomePlanet: from LastName', log)
    m = df['HomePlanet'].isnull() & df['Deck'].isin(DECK_TO_PLANET.keys())
    df.loc[m, 'HomePlanet'] = df.loc[m, 'Deck'].map(DECK_TO_PLANET)
    _log(log, 'HomePlanet: from deck', m.sum())

    # 4. Cabin: side is never mixed within a group; deck/num match the group's
    #    modal cabin 87% of the time
    _fill_from_group(df, 'Side', 'GroupId', 'Side: from group', log)
    _fill_from_group(df, 'Deck', 'GroupId', 'Deck: from group', log)
    _fill_from_group(df, 'Cabin_num', 'GroupId', 'Cabin_num: from group', log)

    # 5. VIP: no Earth passenger is a VIP
    m = df['VIP'].isnull() & (df['HomePlanet'] == 'Earth')
    df.loc[m, 'VIP'] = False
    _log(log, 'VIP: Earth -> False', m.sum())

    return df, log


def fill_remaining(df, log=None):
    """Statistical fallback for whatever the rules could not fill.

    Statistics come from train rows only (``Dataset == 'train'``).
    """
    df = df.copy()
    log = [] if log is None else log
    tr = df[df['Dataset'] == 'train']

    for col in ['HomePlanet', 'Destination', 'Deck', 'Side']:
        m = df[col].isnull()
        df.loc[m, col] = tr[col].mode().iloc[0]
        _log(log, f'{col}: mode', m.sum())

    # Booleans: mode (False for both)
    for col in ['CryoSleep', 'VIP']:
        m = df[col].isnull()
        df.loc[m, col] = tr[col].mode().iloc[0]
        _log(log, f'{col}: mode', m.sum())
    df['CryoSleep'] = df['CryoSleep'].astype(bool)
    df['VIP'] = df['VIP'].astype(bool)

    # Age: median by HomePlanet
    med = tr.groupby('HomePlanet')['Age'].median()
    m = df['Age'].isnull()
    df.loc[m, 'Age'] = df.loc[m, 'HomePlanet'].map(med)
    _log(log, 'Age: median by HomePlanet', m.sum())

    # Spending (only awake adults are left): median of awake adults by HomePlanet
    awake_adults = tr[(tr['CryoSleep'] == False) & (tr['Age'] >= 13)]  # noqa: E712
    n = 0
    for col in SPEND_COLS:
        med = awake_adults.groupby('HomePlanet')[col].median()
        m = df[col].isnull()
        df.loc[m, col] = df.loc[m, 'HomePlanet'].map(med)
        n += m.sum()
    _log(log, 'Spend: awake-adult median by HomePlanet', n)

    # Cabin_num: median within deck
    med = tr.groupby('Deck')['Cabin_num'].median()
    m = df['Cabin_num'].isnull()
    df.loc[m, 'Cabin_num'] = df.loc[m, 'Deck'].map(med)
    _log(log, 'Cabin_num: median by deck', m.sum())

    return df, log


def preprocess(train_df, test_df, verbose=True):
    """Full pipeline. Returns (full_df, imputation_log_df).

    ``full_df`` has every row from train and test (``Dataset`` marks which),
    no missing values in ``FEATURES``, and ``Transported`` as int on train rows.
    """
    full = parse_raw_columns(combine_train_test(train_df, test_df))
    full, log = apply_imputation_rules(full)
    full, log = fill_remaining(full, log)
    full = add_derived_features(full)
    full['CryoSleep_int'] = full['CryoSleep_int'].astype(int)
    full['VIP_int'] = full['VIP_int'].astype(int)

    is_train = full['Dataset'] == 'train'
    full.loc[is_train, TARGET] = full.loc[is_train, TARGET].astype(int)

    remaining = full[FEATURES].isnull().sum()
    assert remaining.sum() == 0, f'Missing values left in features:\n{remaining[remaining > 0]}'

    log_df = pd.DataFrame(log)
    if verbose:
        print(f'Imputed {log_df["Filled"].sum()} values; no missing values left in FEATURES.')
    return full, log_df


def split_xy(full, features=FEATURES):
    """Return X_train, y_train, groups, X_test, test_ids."""
    tr = full[full['Dataset'] == 'train']
    te = full[full['Dataset'] == 'test']
    return (tr[features].reset_index(drop=True),
            tr[TARGET].astype(int).reset_index(drop=True),
            tr['GroupId'].reset_index(drop=True),
            te[features].reset_index(drop=True),
            te['PassengerId'].reset_index(drop=True))
