"""
Helper functions for EDA on the Spaceship Titanic dataset.

Conventions
-----------
- Feature engineering runs on train + test combined (``combine_train_test``) so
  group/family counts see every passenger and the same code can be reused for
  modeling. A ``Dataset`` column marks the origin of each row.
- Every target analysis uses only the train rows (``Transported`` is NaN on test).
- No function mutates its input DataFrame.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy import stats
from sklearn.feature_selection import mutual_info_classif

TARGET = 'Transported'
SPEND_COLS = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
LUXURY_COLS = ['RoomService', 'Spa', 'VRDeck']
BASIC_COLS = ['FoodCourt', 'ShoppingMall']
RAW_CATEGORICAL = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
TARGET_PALETTE = {0: '#d95f02', 1: '#1b9e77'}  # orange = not transported, green = transported


# ---------------------------------------------------------------------------
# Loading & overview
# ---------------------------------------------------------------------------
def combine_train_test(train_df, test_df):
    """Stack train and test with a ``Dataset`` marker column."""
    return pd.concat(
        [train_df.assign(Dataset='train'), test_df.assign(Dataset='test')],
        ignore_index=True,
    )


def train_only(df):
    """Train rows with the target as int (0/1)."""
    out = df[df['Dataset'] == 'train'].copy()
    out[TARGET] = out[TARGET].astype(int)
    return out


def load_and_examine_data(df):
    """Print shape, dtypes, summary statistics, duplicates and missingness."""
    print(f"Dataset Shape: {df.shape}")
    print("\nData Types:")
    print(df.dtypes)
    print("\nSummary Statistics:")
    print(df.describe(include='all').T)
    print(f"\nDuplicate rows (ignoring PassengerId): "
          f"{df.drop(columns='PassengerId').duplicated().sum()}")
    print("\nMissing Values:")
    print(missing_summary(df))
    return df


def missing_summary(df):
    """Missing count and percentage per column, plus share of incomplete rows."""
    missing = df.isnull().sum()
    out = pd.DataFrame({'Missing': missing, 'Percent': (missing / len(df) * 100).round(2)})
    out = out[out['Missing'] > 0].sort_values('Missing', ascending=False)
    print(f"Rows with at least one missing value: {df.isnull().any(axis=1).mean():.1%}")
    return out


# ---------------------------------------------------------------------------
# Missing data
# ---------------------------------------------------------------------------
def visualize_missing_data(df):
    """Missingness matrix and nullity-correlation heatmap."""
    ax = msno.matrix(df, figsize=(14, 5), fontsize=10, sparkline=False)
    ax.set_title('Missing Value Matrix', fontsize=16)
    plt.show()

    ax = msno.heatmap(df, figsize=(10, 7), fontsize=10)
    ax.set_title('Missing Value Correlation Heatmap', fontsize=16)
    plt.show()


def missingness_vs_target(df, target_var=TARGET):
    """Transported rate when each feature is missing vs present.

    A large gap means missingness itself is informative (keep a flag);
    a small gap supports treating it as missing-at-random and imputing.
    """
    rows = []
    for col in df.columns:
        if col in (target_var, 'Dataset') or not df[col].isnull().any():
            continue
        miss = df[col].isnull()
        rows.append({
            'Feature': col,
            'N missing': int(miss.sum()),
            'Rate | missing': df.loc[miss, target_var].mean(),
            'Rate | present': df.loc[~miss, target_var].mean(),
        })
    out = pd.DataFrame(rows).set_index('Feature')
    out['Gap'] = out['Rate | missing'] - out['Rate | present']
    return out.round(3).sort_values('Gap', key=abs, ascending=False)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def feature_engineering(df):
    """Derive features from every raw column.

    - PassengerId -> group features; Cabin -> deck/num/side; Name -> family
    - Age -> age bands; spending -> totals, luxury/basic split, log1p
    - HomePlanet/CryoSleep/Destination/VIP are already clean categoricals:
      they get nullable 0/1 flags and the interactions the EDA flagged.
      One-hot/target encoding is left to the modeling pipeline.

    Intended to be run on the combined train + test frame.
    """
    out = df.copy()

    # Boolean categoricals -> nullable 0/1 (missing stays missing for imputation)
    for col in ['CryoSleep', 'VIP']:
        out[f'{col}_int'] = out[col].map({True: 1, False: 0}).astype('Int64')

    # HomePlanet x Destination route (transported rate ranges 39%-74% across routes)
    out['Route'] = out['HomePlanet'] + '->' + out['Destination']

    # PassengerId -> gggg_pp (group id, position within group)
    pid = out['PassengerId'].str.split('_', expand=True)
    out['GroupId'] = pid[0]
    out['PersonalId'] = pid[1].astype(int)
    out['GroupSize'] = out.groupby('GroupId')['PassengerId'].transform('count')
    out['IsSolo'] = (out['GroupSize'] == 1).astype(int)

    # Cabin -> deck/num/side
    cabin = out['Cabin'].str.split('/', expand=True)
    out['Deck'] = cabin[0]
    out['Cabin_num'] = pd.to_numeric(cabin[1])
    out['Side'] = cabin[2]
    out['Cabin_missing'] = out['Cabin'].isnull().astype(int)
    out['Deck_Side'] = out['Deck'] + '/' + out['Side']
    out['CabinRegion'] = pd.cut(
        out['Cabin_num'], bins=[-1, 300, 600, 900, 1200, 1500, 2000],
        labels=['0-300', '300-600', '600-900', '900-1200', '1200-1500', '1500+'])

    # Name -> family
    names = out['Name'].str.split(' ', n=1, expand=True)
    out['LastName'] = names[1]
    out['FamilySize'] = out.groupby('LastName')['PassengerId'].transform('count')

    # Age bands (children <13 never spend)
    out['AgeGroup'] = pd.cut(
        out['Age'], bins=[-1, 12, 17, 25, 30, 50, 100],
        labels=['0-12', '13-17', '18-25', '26-30', '31-50', '51+'])

    # Spending (NaN treated as 0 for the aggregates)
    spend = out[SPEND_COLS].fillna(0)
    out['TotalSpending'] = spend.sum(axis=1)
    out['LuxurySpending'] = spend[LUXURY_COLS].sum(axis=1)
    out['BasicSpending'] = spend[BASIC_COLS].sum(axis=1)
    out['NumAmenitiesUsed'] = (spend > 0).sum(axis=1)
    out['HasSpent'] = (out['TotalSpending'] > 0).astype(int)
    # Share of spend going to luxury amenities. Non-spenders get 0 (no luxury
    # spend); both they and basic-only spenders are mostly transported.
    out['LuxuryShare'] = (out['LuxurySpending'] / out['TotalSpending']).fillna(0)

    # CryoSleep x Deck: sleepers on A-D/F are ~99% transported, on E/G ~65%
    out['CryoSleep_Deck'] = out['CryoSleep'].map({True: 'Cryo', False: 'Awake'}) + '/' + out['Deck']
    for col in SPEND_COLS + ['TotalSpending', 'LuxurySpending', 'BasicSpending']:
        out[f'log_{col}'] = np.log1p(out[col])

    return out


# ---------------------------------------------------------------------------
# Plot primitives
# ---------------------------------------------------------------------------
def plot_target_rate(df, col, ax=None, target_var=TARGET, order=None):
    """Bar chart of transported rate per category, annotated with counts."""
    ax = ax or plt.gca()
    grp = df.groupby(col, observed=True)[target_var].agg(['mean', 'count'])
    if order is not None:
        grp = grp.reindex([o for o in order if o in grp.index])
    ax.bar(grp.index.astype(str), grp['mean'], color=sns.color_palette('viridis', len(grp)))
    ax.axhline(df[target_var].mean(), color='black', ls='--', lw=1, label='overall')
    for i, (m, n) in enumerate(zip(grp['mean'], grp['count'])):
        ax.text(i, m + 0.01, f'{m:.2f}\n(n={n})', ha='center', va='bottom', fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title(f'{target_var} rate by {col}', fontsize=12)
    ax.set_xlabel(col)
    ax.set_ylabel('Rate')
    ax.tick_params(axis='x', rotation=45)
    return ax


def _grid(n, ncols=3, height=4.5):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, height * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes


# ---------------------------------------------------------------------------
# Target, categorical, numerical
# ---------------------------------------------------------------------------
def analyze_target(df, target_var=TARGET):
    """Target counts and balance."""
    counts = df[target_var].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index.astype(str), counts.values, color=sns.color_palette('viridis', 2))
    for i, v in enumerate(counts.values):
        ax.text(i, v, f'{v} ({v / counts.sum():.1%})', ha='center', va='bottom')
    ax.set_title(f'Distribution of {target_var}', fontsize=14)
    plt.tight_layout()
    plt.show()


def analyze_categorical_vars(df, cols, target_var=TARGET):
    """Transported rate for each categorical feature (missing shown as 'NA')."""
    plot_df = df.copy()
    for col in cols:
        if isinstance(plot_df[col].dtype, pd.CategoricalDtype):
            plot_df[col] = plot_df[col].cat.add_categories('NA').fillna('NA')
        else:
            plot_df[col] = plot_df[col].astype(object).fillna('NA').astype(str)
    fig, axes = _grid(len(cols))
    for ax, col in zip(axes, cols):
        plot_target_rate(plot_df, col, ax=ax, target_var=target_var)
    plt.tight_layout()
    plt.show()


def analyze_numerical_vars(df, cols, target_var=TARGET):
    """Overlaid histograms by target. Use log_ columns for spending."""
    fig, axes = _grid(len(cols))
    for ax, col in zip(axes, cols):
        sns.histplot(data=df, x=col, hue=target_var, palette=TARGET_PALETTE, bins=40, element='step',
                     stat='density', common_norm=False, ax=ax)
        ax.set_title(f'{col} by {target_var}', fontsize=12)
    plt.tight_layout()
    plt.show()


def analyze_correlations(df, cols, target_var=TARGET):
    """Spearman correlation heatmap (robust to the heavy spending skew).

    The target is included so its row/column can be read off directly.
    """
    corr = df[cols + [target_var]].astype(float).corr(method='spearman')
    plt.figure(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, linewidths=0.5, annot_kws={'size': 8})
    plt.title('Spearman Correlation Heatmap', fontsize=16)
    plt.tight_layout()
    plt.show()
    return corr


# ---------------------------------------------------------------------------
# Domain-specific analyses
# ---------------------------------------------------------------------------
def analyze_cryosleep_and_age_rules(df):
    """Check the logical constraints: CryoSleep and children (<13) never spend."""
    cryo = df.dropna(subset=['CryoSleep'])
    print('Passengers with spending > 0, by CryoSleep:')
    print(pd.crosstab(cryo['CryoSleep'], cryo['HasSpent'], margins=True))

    age = df.dropna(subset=['AgeGroup'])
    print('\nShare with spending > 0, by AgeGroup:')
    print(age.groupby('AgeGroup', observed=True)['HasSpent'].mean().round(3))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    plot_target_rate(cryo.assign(CryoSleep=cryo['CryoSleep'].astype(str)), 'CryoSleep', ax=axes[0])
    awake = cryo[cryo['CryoSleep'] == False]  # noqa: E712
    plot_target_rate(awake.assign(HasSpent=awake['HasSpent'].map({0: 'No', 1: 'Yes'})),
                     'HasSpent', ax=axes[1])
    axes[1].set_title('Awake passengers: rate by HasSpent', fontsize=12)
    plt.tight_layout()
    plt.show()


def analyze_age(df, target_var=TARGET):
    """Age distribution by target and transported rate per age band."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    sns.histplot(data=df, x='Age', hue=target_var, palette=TARGET_PALETTE, binwidth=2, element='step',
                 stat='density', common_norm=False, ax=axes[0])
    axes[0].set_title('Age by Transported', fontsize=12)
    plot_target_rate(df, 'AgeGroup', ax=axes[1])
    plt.tight_layout()
    plt.show()


def analyze_spending_patterns(df, target_var=TARGET):
    """Spending on log scale, zero vs non-zero, and luxury vs basic split."""
    fig, axes = _grid(len(SPEND_COLS) + 1)
    for ax, col in zip(axes, SPEND_COLS + ['TotalSpending']):
        nonzero = df[df[col] > 0]
        sns.histplot(data=nonzero, x=f'log_{col}', hue=target_var, palette=TARGET_PALETTE, bins=30, element='step',
                     stat='density', common_norm=False, ax=ax)
        ax.set_title(f'log1p({col}) | spend > 0  (zero: {(df[col] == 0).mean():.0%})',
                     fontsize=11)
    plt.tight_layout()
    plt.show()

    # Rate when each amenity is used vs not used
    rows = []
    for col in SPEND_COLS:
        used = df[col] > 0
        rows.append({'Amenity': col, 'Group': 'Luxury' if col in LUXURY_COLS else 'Basic',
                     'Rate | used': df.loc[used, target_var].mean(),
                     'Rate | not used': df.loc[df[col] == 0, target_var].mean()})
    usage = pd.DataFrame(rows).set_index('Amenity').round(3)
    print(usage)

    fig, ax = plt.subplots(figsize=(7, 6))
    sample = df.sample(min(3000, len(df)), random_state=0)
    sns.scatterplot(data=sample, x='log_BasicSpending', y='log_LuxurySpending',
                    hue=target_var, palette=TARGET_PALETTE, alpha=0.5, s=15, ax=ax)
    ax.set_title('Luxury vs Basic spending (log1p)', fontsize=12)
    plt.tight_layout()
    plt.show()
    return usage


def analyze_cabin(df, target_var=TARGET):
    """Deck, side and cabin-number position vs target."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    plot_target_rate(df, 'Deck', ax=axes[0], order=list('ABCDEFGT'))
    plot_target_rate(df, 'Side', ax=axes[1])
    plot_target_rate(df, 'CabinRegion', ax=axes[2])
    plt.tight_layout()
    plt.show()

    # Rolling transported rate along the ship, per side
    fig, ax = plt.subplots(figsize=(15, 4.5))
    for side, grp in df.dropna(subset=['Cabin_num']).sort_values('Cabin_num').groupby('Side'):
        roll = grp[target_var].rolling(150, center=True, min_periods=50).mean()
        ax.plot(grp['Cabin_num'], roll, label=f'Side {side}')
    ax.axhline(df[target_var].mean(), color='black', ls='--', lw=1)
    ax.set_xlabel('Cabin_num')
    ax.set_ylabel('Rolling rate (window=150)')
    ax.set_title('Transported rate along the ship', fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Deck x Side interaction
    pivot = df.pivot_table(index='Deck', columns='Side', values=target_var, aggfunc='mean')
    plt.figure(figsize=(5, 5))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='viridis', vmin=0, vmax=1)
    plt.title('Transported rate: Deck x Side', fontsize=12)
    plt.tight_layout()
    plt.show()


def analyze_group_patterns(df, target_var=TARGET):
    """Group size, solo travel, family size, and within-group target agreement."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    plot_target_rate(df, 'GroupSize', ax=axes[0])
    plot_target_rate(df.assign(IsSolo=df['IsSolo'].map({0: 'Group', 1: 'Solo'})),
                     'IsSolo', ax=axes[1])
    fam = df.dropna(subset=['FamilySize']).copy()
    fam['FamilySize'] = fam['FamilySize'].clip(upper=10).astype(int)
    plot_target_rate(fam, 'FamilySize', ax=axes[2])
    axes[2].set_title(f'{target_var} rate by FamilySize (capped at 10)', fontsize=12)
    plt.tight_layout()
    plt.show()

    multi = df[df['GroupSize'] > 1]
    group_mean = multi.groupby('GroupId')[target_var].mean()
    status = pd.cut(group_mean, bins=[-0.01, 0.01, 0.99, 1.01],
                    labels=['None transported', 'Mixed', 'All transported'])
    print('Multi-passenger groups by outcome agreement:')
    print(status.value_counts(normalize=True).round(3))

    # How predictive is the leave-one-out group mean?
    g = multi.groupby('GroupId')[target_var]
    loo = (g.transform('sum') - multi[target_var]) / (g.transform('count') - 1)
    agree = ((loo > 0.5) == multi[target_var].astype(bool))[loo != 0.5].mean()
    print(f'\nLeave-one-out group majority matches passenger outcome: {agree:.1%} '
          f'(baseline {max(df[target_var].mean(), 1 - df[target_var].mean()):.1%})')


def imputation_consistency(df):
    """Quantify the deterministic relationships usable for imputation (train + test)."""
    report = {}

    hp = df.dropna(subset=['HomePlanet']).groupby('GroupId')['HomePlanet'].nunique()
    report['Groups with >1 HomePlanet'] = int((hp > 1).sum())
    known_hp = df.dropna(subset=['HomePlanet']).groupby('GroupId')['HomePlanet'].first()
    miss_hp = df[df['HomePlanet'].isnull()]
    report['Missing HomePlanet recoverable from group'] = (
        f"{miss_hp['GroupId'].isin(known_hp.index).sum()} / {len(miss_hp)}")

    ln = df.dropna(subset=['HomePlanet', 'LastName']).groupby('LastName')['HomePlanet'].nunique()
    report['Families (LastName) with >1 HomePlanet'] = int((ln > 1).sum())

    side = df.dropna(subset=['Side']).groupby('GroupId')['Side'].nunique()
    report['Groups with >1 Side'] = int((side > 1).sum())
    miss_side = df[df['Cabin'].isnull()]
    report['Missing Cabin with a groupmate whose cabin is known'] = (
        f"{miss_side['GroupId'].isin(df.dropna(subset=['Cabin'])['GroupId']).sum()} "
        f"/ {len(miss_side)}")

    cryo_true = df['CryoSleep'] == True  # noqa: E712
    report['CryoSleep=True with any spend'] = int((cryo_true & (df['TotalSpending'] > 0)).sum())
    report['Missing spend values on CryoSleep=True rows (-> 0)'] = int(
        df.loc[cryo_true, SPEND_COLS].isnull().sum().sum())
    report['Missing CryoSleep with spend > 0 (-> False)'] = int(
        (df['CryoSleep'].isnull() & (df['TotalSpending'] > 0)).sum())
    report['Age < 13 with any spend'] = int(((df['Age'] < 13) & (df['TotalSpending'] > 0)).sum())

    for k, v in report.items():
        print(f'{k:<55} {v}')

    plt.figure(figsize=(6, 5))
    sns.heatmap(pd.crosstab(df['Deck'], df['HomePlanet']), annot=True, fmt='d', cmap='viridis')
    plt.title('Deck x HomePlanet (counts, train + test)', fontsize=12)
    plt.tight_layout()
    plt.show()
    return report


# ---------------------------------------------------------------------------
# Train vs test
# ---------------------------------------------------------------------------
def compare_train_test(df, numeric_cols, categorical_cols):
    """Distribution shift check: KS test for numerics, proportions for categoricals."""
    tr, te = df[df['Dataset'] == 'train'], df[df['Dataset'] == 'test']
    rows = []
    for col in numeric_cols:
        ks = stats.ks_2samp(tr[col].dropna(), te[col].dropna())
        rows.append({'Feature': col, 'Train mean': tr[col].mean(), 'Test mean': te[col].mean(),
                     'KS stat': ks.statistic, 'p-value': ks.pvalue})
    print(pd.DataFrame(rows).set_index('Feature').round(3))

    fig, axes = _grid(len(categorical_cols))
    for ax, col in zip(axes, categorical_cols):
        props = (df.assign(**{col: df[col].astype(object).fillna('NA').astype(str)})
                   .groupby('Dataset')[col].value_counts(normalize=True)
                   .rename('Proportion').reset_index())
        sns.barplot(data=props, x=col, y='Proportion', hue='Dataset', ax=ax)
        ax.set_title(f'{col}: train vs test', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Feature ranking
# ---------------------------------------------------------------------------
def cramers_v(x, y):
    """Bias-corrected Cramér's V between two categorical series."""
    table = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.values.sum()
    r, k = table.shape
    phi2 = max(0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    r_corr, k_corr = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return np.sqrt(phi2 / max(1e-12, min(k_corr - 1, r_corr - 1)))


def feature_ranking(df, categorical_cols, numeric_cols, target_var=TARGET, random_state=0):
    """Rank features by mutual information (all) and Cramér's V (categoricals)."""
    X = pd.DataFrame(index=df.index)
    for col in categorical_cols:
        X[col] = pd.factorize(df[col].astype(object).fillna('NA'))[0]
    for col in numeric_cols:
        X[col] = df[col].fillna(df[col].median())
    discrete = [c in categorical_cols for c in X.columns]
    mi = mutual_info_classif(X, df[target_var], discrete_features=discrete,
                             random_state=random_state)
    out = pd.DataFrame({'MutualInfo': mi}, index=X.columns)
    out['Type'] = ['categorical' if d else 'numeric' for d in discrete]
    out['CramersV'] = [cramers_v(df[c].astype(object).fillna('NA'), df[target_var])
                       if c in categorical_cols else np.nan for c in X.columns]
    out = out.sort_values('MutualInfo', ascending=False).round(4)

    plt.figure(figsize=(10, 0.35 * len(out) + 1))
    sns.barplot(x=out['MutualInfo'], y=out.index, hue=out['Type'], dodge=False)
    plt.title('Mutual information with Transported', fontsize=14)
    plt.tight_layout()
    plt.show()
    return out


# ---------------------------------------------------------------------------
# Deeper dives (interactions surfaced after the first EDA pass)
# ---------------------------------------------------------------------------
def analyze_cryosleep_by_deck(df, target_var=TARGET):
    """Transported rate and counts for CryoSleep x Deck."""
    d = df.dropna(subset=['CryoSleep', 'Deck']).copy()
    d['CryoSleep'] = d['CryoSleep'].astype(bool)
    rate = d.pivot_table(index='CryoSleep', columns='Deck', values=target_var, aggfunc='mean')
    count = d.pivot_table(index='CryoSleep', columns='Deck', values=target_var, aggfunc='count')
    annot = rate.round(2).astype(str) + '\n(n=' + count.fillna(0).astype(int).astype(str) + ')'

    plt.figure(figsize=(12, 3.8))
    sns.heatmap(rate, annot=annot, fmt='', cmap='viridis', vmin=0, vmax=1,
                annot_kws={'size': 9})
    plt.title('Transported rate: CryoSleep x Deck', fontsize=13)
    plt.tight_layout()
    plt.show()
    return rate.round(3)


def analyze_luxury_share(df, target_var=TARGET, random_state=0):
    """Transported rate by share of spend on luxury amenities (spenders only)."""
    sp = df[df['TotalSpending'] > 0].copy()
    bins = [-0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    sp['LuxuryShareBin'] = pd.cut(sp['LuxuryShare'], bins=bins,
                                  labels=['0-0.1', '0.1-0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9', '0.9-1'])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    plot_target_rate(sp, 'LuxuryShareBin', ax=ax)
    ax.set_title('Spenders: transported rate by LuxuryShare', fontsize=12)
    plt.tight_layout()
    plt.show()

    X = sp[['LuxuryShare', 'log_TotalSpending', 'log_LuxurySpending', 'log_BasicSpending']]
    mi = mutual_info_classif(X, sp[target_var], random_state=random_state)
    out = pd.Series(mi, index=X.columns, name='MutualInfo (spenders only)')
    return out.sort_values(ascending=False).round(3)


def analyze_cabin_within_deck(df, decks=('F', 'G'), target_var=TARGET, window=100):
    """Cabin_num ranges per deck, and rolling rate along the ship within each deck."""
    print('Cabin_num range by deck:')
    print(df.groupby('Deck')['Cabin_num'].agg(['min', 'max', 'count']).T)

    fig, axes = plt.subplots(1, len(decks), figsize=(8 * len(decks), 4.5), sharey=True)
    for ax, deck in zip(np.atleast_1d(axes), decks):
        d = df[df['Deck'] == deck].sort_values('Cabin_num')
        for side, grp in d.groupby('Side'):
            roll = grp[target_var].rolling(window, center=True, min_periods=window // 3).mean()
            ax.plot(grp['Cabin_num'], roll, label=f'Side {side}')
        ax.axhline(df[target_var].mean(), color='black', ls='--', lw=1)
        ax.axvspan(700, 1100, color='grey', alpha=0.15)
        ax.set_title(f'Deck {deck}: rolling rate (window={window})', fontsize=12)
        ax.set_xlabel('Cabin_num')
        ax.legend()
    np.atleast_1d(axes)[0].set_ylabel('Rate')
    plt.tight_layout()
    plt.show()


def analyze_child_age(df, target_var=TARGET):
    """Fine-grained transported rate and spending share for under-18s."""
    d = df.dropna(subset=['Age']).copy()
    d['AgeDetail'] = pd.cut(d['Age'], bins=[-1, 0, 4, 8, 12, 17, 100],
                            labels=['0', '1-4', '5-8', '9-12', '13-17', '18+'])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    plot_target_rate(d, 'AgeDetail', ax=ax)
    plt.tight_layout()
    plt.show()
    return d.groupby('AgeDetail', observed=True).agg(
        Transported=(target_var, 'mean'), HasSpent=('HasSpent', 'mean'), N=(target_var, 'size')).round(3)


def additional_imputation_checks(full, train):
    """Extra imputation evidence: CryoSleep from zero spend, VIP by planet,
    CryoSleep agreement in groups, and groupmate outcome correlation."""
    z = full[(full['TotalSpending'] == 0) & (full['Age'] >= 13)].dropna(subset=['CryoSleep'])
    print(f"P(CryoSleep=True | zero spend, age>=13): {z['CryoSleep'].astype(bool).mean():.3f}")
    print(f"Missing CryoSleep with zero spend: "
          f"{int((full['CryoSleep'].isnull() & (full['TotalSpending'] == 0)).sum())}")
    aw0 = train[(train['CryoSleep'] == False) & (train['TotalSpending'] == 0)]  # noqa: E712
    print(f"Awake & zero spend: n={len(aw0)}, share age<13={(aw0['Age'] < 13).mean():.2f}")

    print('\nVIP by HomePlanet (train + test):')
    print(pd.crosstab(full['HomePlanet'], full['VIP']))

    multi = full[full['GroupSize'] > 1].dropna(subset=['CryoSleep'])
    mixed = (multi.groupby('GroupId')['CryoSleep'].nunique() > 1).mean()
    print(f'\nMulti-passenger groups with mixed CryoSleep: {mixed:.1%}')

    g = train[train['GroupSize'] > 1]
    gm = g.groupby('GroupId')[TARGET]
    loo = (gm.transform('sum') - g[TARGET]) / (gm.transform('count') - 1)
    print(f'Correlation of groupmates\' mean outcome with own outcome: {np.corrcoef(loo, g[TARGET])[0, 1]:.3f}')
