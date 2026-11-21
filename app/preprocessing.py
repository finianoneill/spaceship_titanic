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


def fill_remaining(df, log=None, spend_fill='median'):
    """Statistical fallback for whatever the rules could not fill.

    Statistics come from train rows only (``Dataset == 'train'``).

    ``spend_fill`` controls the spending columns left after the rules:
    'median' (awake-adult median by HomePlanet), 'zero', or 'nan' (leave
    missing; only for models that handle NaN natively, e.g. CatBoost).
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

    # Spending (only awake adults are left)
    n = int(df[SPEND_COLS].isnull().sum().sum())
    if spend_fill == 'median':
        awake_adults = tr[(tr['CryoSleep'] == False) & (tr['Age'] >= 13)]  # noqa: E712
        for col in SPEND_COLS:
            med = awake_adults.groupby('HomePlanet')[col].median()
            m = df[col].isnull()
            df.loc[m, col] = df.loc[m, 'HomePlanet'].map(med)
        _log(log, 'Spend: awake-adult median by HomePlanet', n)
    elif spend_fill == 'zero':
        df[SPEND_COLS] = df[SPEND_COLS].fillna(0)
        _log(log, 'Spend: 0', n)
    elif spend_fill == 'nan':
        _log(log, 'Spend: left as NaN (aggregates use known values)', 0)
    else:
        raise ValueError(f'Unknown spend_fill: {spend_fill}')

    # Cabin_num: median within deck
    med = tr.groupby('Deck')['Cabin_num'].median()
    m = df['Cabin_num'].isnull()
    df.loc[m, 'Cabin_num'] = df.loc[m, 'Deck'].map(med)
    _log(log, 'Cabin_num: median by deck', m.sum())

    return df, log


def preprocess(train_df, test_df, spend_fill='median', verbose=True):
    """Full pipeline. Returns (full_df, imputation_log_df).

    ``full_df`` has every row from train and test (``Dataset`` marks which),
    no missing values in ``FEATURES``, and ``Transported`` as int on train rows.
    """
    full = parse_raw_columns(combine_train_test(train_df, test_df))
    full, log = apply_imputation_rules(full)
    full, log = fill_remaining(full, log, spend_fill=spend_fill)
    full = add_derived_features(full)
    full['CryoSleep_int'] = full['CryoSleep_int'].astype(int)
    full['VIP_int'] = full['VIP_int'].astype(int)

    is_train = full['Dataset'] == 'train'
    full.loc[is_train, TARGET] = full.loc[is_train, TARGET].astype(int)

    remaining = full[FEATURES].isnull().sum()
    if spend_fill == 'nan':
        remaining = remaining.drop([f'log_{c}' for c in SPEND_COLS])
    assert remaining.sum() == 0, f'Missing values left in features:\n{remaining[remaining > 0]}'

    log_df = pd.DataFrame(log)
    if verbose:
        print(f'[spend_fill={spend_fill}] Imputed {log_df["Filled"].sum()} values; '
              f'no unexpected missing values in FEATURES.')
    return full, log_df


def split_xy(full, features=FEATURES):
    """Return X_train, y_train, groups, X_test, test_ids.

    Pass ``features + ['GroupId']`` when a fold-aware feature (e.g. the cabin
    neighbour rate) needs the group id; the model wrapper drops it.
    """
    tr = full[full['Dataset'] == 'train']
    te = full[full['Dataset'] == 'test']
    return (tr[features].reset_index(drop=True),
            tr[TARGET].astype(int).reset_index(drop=True),
            tr['GroupId'].reset_index(drop=True),
            te[features].reset_index(drop=True),
            te['PassengerId'].reset_index(drop=True))


# ---------------------------------------------------------------------------
# Round 3: group-level and cabin-neighbourhood features (no target used)
# ---------------------------------------------------------------------------
GROUP_FEATURES = [
    'Grp_CryoShare', 'Grp_LogSpendMean', 'Grp_LuxuryShareMean', 'Grp_HasChild',
    'Grp_NumCabins', 'Grp_SameCabin', 'Grp_SameLastName', 'Grp_AgeGap', 'IsChild',
]
NEIGHBORHOOD_FEATURES = ['Cabin_Occupancy', 'Nbr_CryoShare']


def _others_mean(df, key, col):
    """Mean of ``col`` over the *other* rows sharing ``key`` (NaN if none)."""
    g = df.groupby(key)[col]
    s, n = g.transform('sum'), g.transform('count')
    return ((s - df[col]) / (n - 1)).where(n > 1)


def add_group_features(full, nbr_k=10, nbr_prior=5.0):
    """Features describing each passenger's groupmates and cabin neighbours.

    Run on the imputed, derived train + test frame (output of ``preprocess``).
    Uses no labels, so combining train and test is legitimate.
    """
    from app.modeling import neighbor_rate  # local import avoids a cycle

    df = full.copy()
    df['IsChild'] = (df['Age'] < 13).astype(int)
    # Cabin identity only where it is real: the raw Cabin, or the groupmate-modal
    # cabin (rule 4). Median-filled cabin numbers are not real cabins.
    from_group = df['Cabin'].isnull() & df['GroupId'].isin(df.loc[df['Cabin'].notnull(), 'GroupId'])
    key = df['Deck'] + '/' + df['Cabin_num'].astype(int).astype(str) + '/' + df['Side']
    df['CabinKey'] = key.where(df['Cabin'].notnull() | from_group)

    df['Grp_CryoShare'] = _others_mean(df, 'GroupId', 'CryoSleep_int')
    df['Grp_LogSpendMean'] = _others_mean(df, 'GroupId', 'log_TotalSpending')
    df['Grp_LuxuryShareMean'] = _others_mean(df, 'GroupId', 'LuxuryShare')
    df['Grp_AgeGap'] = df['Age'] - _others_mean(df, 'GroupId', 'Age')
    kids = df.groupby('GroupId')['IsChild'].transform('sum') - df['IsChild']
    df['Grp_HasChild'] = (kids > 0).astype(int)
    df['Grp_NumCabins'] = df.groupby('GroupId')['CabinKey'].transform('nunique')
    df['Grp_SameCabin'] = (df.groupby(['GroupId', 'CabinKey'])['PassengerId']
                             .transform('count') - 1)  # NaN where the cabin is unknown
    same_ln = df.groupby(['GroupId', 'LastName'])['PassengerId'].transform('count') - 1
    df['Grp_SameLastName'] = same_ln.fillna(0).astype(int)

    df['Cabin_Occupancy'] = df.groupby('CabinKey')['PassengerId'].transform('count')
    df['Nbr_CryoShare'] = neighbor_rate(df, df, df['CryoSleep_int'], k=nbr_k, prior=nbr_prior,
                                        exclude_own_group=True)
    return df.drop(columns='CabinKey')
