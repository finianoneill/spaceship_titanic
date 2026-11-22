# Spaceship Titanic

My work on the Kaggle [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) competition: predict which passengers were transported to another dimension. It's a balanced binary classification problem scored on accuracy.

The project has two stages:
1. **EDA** (`eda.ipynb`): find the structure in the data, and the rules that make imputation precise.
2. **Modeling** (`model.ipynb`): turn those findings into features and validate them with grouped, multi-seed cross-validation.

Reusable code lives in `app/`, so the notebooks only record analysis and experiments.

## Results

| Version | Model | CV accuracy | Public LB |
|---|---|---|---|
| Round 1 | CatBoost, 24 features, default parameters | 0.8164 (1 split); **0.8169 ± 0.0012** (5 seeds) | 0.80266 |
| Round 2 | Best: CatBoost + XGBoost ensemble | 0.8174 (5 seeds, +0.05 pts) | not submitted |
| Round 3 | Best: unchanged (group features, neighborhood features, pseudo-labels all rejected) | 0.8169 (5 seeds, ±0.00) | not submitted |
| **Round 4** | **TabPFN v3.5 foundation model, 24 features** | **0.8193 ± 0.0015** (5 seeds, +0.24 pts, better on 5/5) | **0.81458** (rank ~126) |

**Rounds 2–3 confirmed a plateau at about 0.817 grouped-CV accuracy.** Nothing beat round 1 by the +0.3-point bar for submitting:
- **Round 2 (more modeling):** spending-fill variants, a cabin-neighbor target feature, LightGBM/XGBoost ensembles, and Optuna. The boosting models' predictions correlate at 0.98–0.99, and tuning overfit its own split.
- **Round 3 (new information):** group features, cabin-neighborhood features, and pseudo-labeling. The group signals are already captured by existing features, and pseudo-labels only reinforce cases the model already gets right.

**Round 4 (a foundation model):** [TabPFN](https://github.com/PriorLabs/TabPFN) v3.5, a transformer pretrained on synthetic tabular data, is the first model to beat CatBoost on every paired seed. The gain (+0.24 points) is just under the submission bar, but TabPFN is submitted as a documented exception because the gain is consistent. Blending it with CatBoost (predictions correlate 0.98) and fine-tuning it (seed-0 pilot: +0.08 points, correlation 0.999 with the frozen model) both failed to add anything. A different model family landing at the same place is further evidence that about 0.82 is this data's ceiling.

On the public leaderboard it scored **0.81458**, up from 0.80266 (+1.19 points, from rank ~1,060 to ~126). The two submissions disagree on 265 test passengers, and TabPFN is right on 158 of them versus 107 for CatBoost (sign test z ≈ 3.1), so the gain is unlikely to be luck. The gap between CV and the leaderboard also shrank from 1.4 to 0.5 points.

The final model is TabPFN v3.5. Details are in sections 18, 22 and 26 of `model.ipynb`.

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
│   ├── modeling.py         # grouped CV, model factories, neighbor feature, ensembling, Optuna, submissions
│   └── run_tabpfn.py       # TabPFN CV + test predictions in a separate process, cached to artifacts/
├── eda.ipynb               # exploratory analysis (outputs committed)
├── model.ipynb             # modeling experiments (outputs stripped on commit)
├── data/                   # competition data (git-ignored)
├── artifacts/              # cached TabPFN / CatBoost predictions (git-ignored)
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

**TabPFN licence:** the TabPFN v3.5 weights are free for non-commercial use (research and evaluation) only. On the first run, TabPFN asks you to accept the licence at Prior Labs (https://ux.priorlabs.ai) and then downloads the weights. Treat the API key it gives you as a secret.

### Data

You need a Kaggle API token in `~/.kaggle/access_token` (or the legacy `~/.kaggle/kaggle.json`), and you must have joined the competition on Kaggle.

```bash
kaggle competitions download -c spaceship-titanic -p data
unzip data/spaceship-titanic.zip -d data
```

### Running

First compute the TabPFN predictions that round 4 of `model.ipynb` loads. They run in a separate process because PyTorch's and CatBoost's OpenMP runtimes crash when used in the same process on macOS:

```bash
python -m app.run_tabpfn --features full                  # ~10 min on an M-series GPU: 5 seeds + test fits
python -m app.run_tabpfn --features compact --no-test
python -m app.run_tabpfn --finetune --seeds 0 --no-test   # optional fine-tuning pilot, ~1-2 hours
```

Then:

```bash
jupyter lab   # then run eda.ipynb and model.ipynb top to bottom
```

Or run a notebook from start to finish without opening it:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 model.ipynb
```

`model.ipynb` takes about 50 minutes on a laptop, mostly the multi-seed CatBoost runs, Optuna and pseudo-labeling. Round 4 only reads cached predictions, so it takes seconds.

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

`model.ipynb` writes `submissions/<model>_cv<score>.csv`. The current best is `submissions/tabpfn_v35_cv0.8193.csv`. To submit:

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
- [Prior Labs](https://priorlabs.ai) for TabPFN

## Citation

Addison Howard, Ashley Chow, and Ryan Holbrook. *Spaceship Titanic*. Kaggle, 2022. https://www.kaggle.com/competitions/spaceship-titanic

```bibtex
@misc{spaceship-titanic,
    author = {Addison Howard and Ashley Chow and Ryan Holbrook},
    title = {Spaceship Titanic},
    year = {2022},
    howpublished = {\url{https://www.kaggle.com/competitions/spaceship-titanic}},
    note = {Kaggle}
}
```
