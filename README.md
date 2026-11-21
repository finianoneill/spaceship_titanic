# Spaceship Titanic

My work on the Kaggle [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) competition: predict which passengers were transported to another dimension. It's a balanced binary classification problem scored on accuracy.

The project has two stages:
1. **EDA** (`eda.ipynb`): find the structure in the data, and the rules that make imputation precise.
2. **Modeling** (`model.ipynb`): turn those findings into features and validate them with grouped, multi-seed cross-validation.

Reusable code lives in `app/`, so the notebooks only record analysis and experiments.

## Results

| Version | Model | CV accuracy | Public LB |
|---|---|---|---|
| Round 1 | CatBoost, 24 features, default parameters | 0.8164 (1 split) | 0.80266 |
| Round 2 | *in progress*: imputation and neighbor-feature ablations, LightGBM/XGBoost ensemble, Optuna | – | – |

**Leaderboard context:** the public board is very compressed. About 500 of roughly 2,200 teams sit between 0.800 and 0.805, ≥ 0.81 is the top ~11%, and ≥ 0.82 the top ~5%. The public score's standard error is about ±0.8 points, so decisions in this repo are made on CV, not leaderboard rank. The few scores above 0.93 are far beyond what the label noise allows and aren't treated as reachable benchmarks.

## Key findings (from the EDA)

- **What passengers spend on matters, not how much.** `LuxuryShare` (RoomService + Spa + VRDeck as a share of total spending) takes spenders from 71% transported down to 11%. It's the strongest single feature.
- **CryoSleep × Deck:** sleepers on decks A–D and F are 94–99% transported, but only about 65% on decks E and G.
- **Cabin position only makes sense within a deck:** on deck G, side S, cabins 800–1,000 reach about 95% transported (rolling rate), while side P stays flat.
- **Deterministic rules for imputation:** sleepers and children under 13 never spend; groups never mix HomePlanet or cabin side; decks A, B, C and T are Europa only and G is Earth only; no Earth passenger is a VIP.
- **Train and test come from the same distribution** (all KS p-values > 0.18), and no group spans both sets.

See section 15 of `eda.ipynb` for the full list with numbers.

## Project structure

```
.
├── app/
│   ├── eda_helpers.py      # EDA plots/analyses + feature engineering (parse_raw_columns, add_derived_features)
│   ├── preprocessing.py    # imputation rules, statistical fill, feature lists, preprocess()
│   └── modeling.py         # grouped CV, model factories, neighbor feature, ensembling, Optuna, submissions
├── eda.ipynb               # exploratory analysis (outputs committed)
├── model.ipynb             # modeling experiments (outputs stripped on commit)
├── data/                   # competition data (git-ignored)
├── submissions/            # generated submission CSVs (git-ignored)
└── requirements.txt
```

## Getting started

### Installation

```bash
git clone https://github.com/finianoneill/spaceship_titanic.git
cd spaceship_titanic

conda create -n spaceship-titanic python=3.12
conda activate spaceship-titanic
pip install -r requirements.txt

# macOS only: XGBoost/LightGBM need the OpenMP runtime
conda install -c conda-forge llvm-openmp

# Strip model.ipynb outputs on commit (eda.ipynb is excluded in .gitattributes)
nbstripout --install
```

### Data

You need a Kaggle API token in `~/.kaggle/access_token` (or the legacy `~/.kaggle/kaggle.json`), and you must have joined the competition on Kaggle.

```bash
kaggle competitions download -c spaceship-titanic -p data
unzip data/spaceship-titanic.zip -d data
```

### Running

```bash
jupyter lab   # then run eda.ipynb and model.ipynb top to bottom
```

Or run a notebook from start to finish without opening it:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 model.ipynb
```

`model.ipynb` takes roughly 25 minutes on a laptop, mostly the multi-seed CatBoost runs and Optuna.

## Data processing

`app.preprocessing.preprocess(train_df, test_df)` runs on train and test combined, so group and family counts see every passenger:

1. **Parse** `PassengerId` → group and position in group, `Cabin` → deck / number / side, `Name` → last name.
2. **Deterministic imputation rules**, applied in order and logged with counts:
   - CryoSleep from spending (spends → awake; zero spending at age ≥ 13 → asleep, which is right 96% of the time)
   - spending = 0 for sleepers and children under 13
   - HomePlanet from groupmates → last name → single-planet decks
   - cabin side, deck and number from groupmates
   - VIP = False for Earth passengers
3. **Statistical fallback** for whatever is left: mode or median, using statistics from train rows only. The spending fill is configurable (`spend_fill='median' | 'zero' | 'nan'`).
4. **Derived features** (`add_derived_features`).

The deterministic rules never use the target, so no label information leaks from preprocessing.

## Features

24 model features (`pp.FEATURES`):

- **Categorical:** `HomePlanet`, `Destination`, `Deck`, `Side`, `Deck_Side`, `Route` (HomePlanet→Destination), `CryoSleep_Deck`
- **Numeric:** `CryoSleep_int`, `VIP_int`, `Age`, `Cabin_num`, `GroupSize`, `IsSolo`, log1p of each amenity and of total/luxury/basic spending, `LuxuryShare`, `HasSpent`, `NumAmenitiesUsed`

Optional fold-aware feature: **`NeighborRate`**, the smoothed transported rate of passengers within ±k cabins on the same deck and side, excluding the passenger's own group. It's computed inside each CV fold from training-fold labels only (`mdl.with_neighbor_feature`).

## Validation

- **`StratifiedGroupKFold` on `GroupId`** (5 folds): groupmates have correlated outcomes and never straddle a split, so CV isn't inflated.
- **Multi-seed, paired comparisons:** every experiment runs on the same 5 split seeds and is compared seed by seed against the baseline. A single 5-fold split moves CV by several tenths of a point on its own.
- **Ablations** confirm each engineered feature. Dropping the four interaction features costs 0.45 points and lowers accuracy in 5 of 5 folds.
- **Tuning** (Optuna) uses its own split seed, separate from the evaluation seeds, so the reported tuned score isn't flattered by selection.

## Making predictions

`model.ipynb` writes `submissions/<model>_cv<score>.csv`. To submit:

```bash
kaggle competitions submit -c spaceship-titanic -f submissions/<file>.csv -m "<description, CV score>"
kaggle competitions submissions -c spaceship-titanic   # check the public score
```

## Author

- Finian O'Neill

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- Kaggle for the competition, dataset and platform
