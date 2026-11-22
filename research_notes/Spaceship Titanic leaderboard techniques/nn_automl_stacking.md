# Neural Nets, AutoML, Stacking and Threshold Calibration on Spaceship Titanic

Context: our CatBoost scores 0.8169 OOF (group-aware CV, 5 seeds) and 0.80266 public LB. Research done 2026-11-21. Most Kaggle-specific numbers come from single notebooks or repos, are self-reported, and are public-LB readings (public LB is about 50% of the test set, so it is noisy). Kaggle notebook pages could not be fetched directly; their scores come from search-result snippets and are flagged as such.

## Q1. AutoML on Spaceship Titanic (AutoGluon, H2O, LightAutoML, FLAML, TPOT, PyCaret)

### Takeaway
Reported AutoML results on Spaceship Titanic land at about 0.79-0.81 public LB. That is roughly where a well-built single CatBoost lands, and I found no evidence that AutoML gets past the ~0.81 public "wall". I found no Spaceship Titanic results at all for LightAutoML, FLAML or TPOT.

### Cited Findings
- AutoGluon: Kaggle notebook "autogluon Spaceship Titanic" (by cozyhn) shows a public score of 0.80944, a best of 0.81108 (version 15), and a 10h 51m runtime. The preset (best_quality or not), the AutoGluon version and the CV score could not be verified, because the page did not render on fetch. Single source, snippet only. — [Kaggle: cozyhn/autogluon-spaceship-titanic](https://www.kaggle.com/code/cozyhn/autogluon-spaceship-titanic)
- For comparison, AutoGluon on the *original* Titanic ("Top 3% in 1 line of code w/ AutoGluon", by Innixma, an AutoGluon author) scored 0.80382. That is a different competition. — [Kaggle: innixma](https://www.kaggle.com/code/innixma/top-3-in-1-line-of-code-w-autogluon)
- H2O AutoML: a stacked ensemble from H2O AutoML reports a Kaggle score "> 0.79". The README gives no H2O version, no CV score, and does not say whether the score is public or private. — [GitHub: vytautas-bunevicius/kaggle-spaceship-titanic](https://github.com/vytautas-bunevicius/kaggle-spaceship-titanic)
- PyCaret: search snippets show two Spaceship Titanic PyCaret notebooks at public 0.81248 and 0.79541. Which score belongs to which notebook could not be confirmed; the candidates are listed here. Unverified. — [Kaggle: lunakim21/spaceship-titanic-pycaret-o](https://www.kaggle.com/code/lunakim21/spaceship-titanic-pycaret-o); [Kaggle: hasanbasriakcay pycaret-model-comparisons](https://www.kaggle.com/code/hasanbasriakcay/spaceship-titanic-pycaret-model-comparisons); [Kaggle: rhythmcam/automl-pycaret-lightgbm](https://www.kaggle.com/code/rhythmcam/automl-pycaret-lightgbm)
- General benchmark: the TabPFN-2.5 report (Nov 2025) says TabPFN-2.5 matches "AutoGluon 1.4, a complex four-hour tuned ensemble" on TabArena. AutoGluon 1.4 is thus the reference strong ensemble in current benchmarks. — [arXiv 2511.08667](https://arxiv.org/pdf/2511.08667v2)

### Inferences
- An AutoGluon public score of 0.809-0.811 is only about 0.007-0.008 above our 0.80266, and the CV-to-LB gap for honest CatBoost runs elsewhere is about 0.005-0.011 (see Q4). AutoGluon (which runs CatBoost/LightGBM/XGBoost/NN/kNN/RF plus stacking) is therefore probably in the same plateau. With newer versions (1.2+), which bundle TabPFNv2/TabM/RealMLP-type models in "extreme"-style presets, it might gain more. That last point is untested here.
- Part of our LB gap may come from our own LB variance or our CV setup, not the model family. Other honest CatBoost pipelines reach public 0.806-0.811 (Q4).

### Gaps
- No Spaceship Titanic scores were found for LightAutoML, FLAML or TPOT.
- The AutoGluon preset and version for the 0.809 notebook are unverified. No AutoGluon OOF on group-aware CV was found.

## Q2. Neural nets (MLP, TabNet, FT-Transformer, TabPFN v1/v2/v2.5/v3.5)

### Takeaway
Plain MLPs and TabNet score 0.79-0.81 public, which is no better than GBDTs. The one clearly positive, well-documented result is **TabPFN v3.5**. On identical group-aware folds it beat CatBoost by about +0.008 OOF (0.825 vs 0.817) and reached public LB 0.8179. A stack of TabPFN variants reached 0.8202 public (0.8264 best reading). Older TabPFN versions (v2.5) and other deep tabular models (RealMLP, TabM, TabR) were *worse* than CatBoost in that same study.

### Cited Findings
- **TaeyanG4 experiments (Oct 2026, single source, self-reported, most findings marked "PENDING cross-host review")**, all on StratifiedGroupKFold grouped by travel group:
  - Honest CatBoost (early stopping on inner CV, not on the scored fold): OOF about 0.8174 (0.8159/0.8178/0.8185 over 3 seeds); public LB 0.81131. The earlier CatBoost with early stopping on the scored fold had OOF 0.8188 and LB 0.80780. — [discoveries.md](https://raw.githubusercontent.com/TaeyanG4/kaggle-spaceship-titanic-experiments/main/discoveries.md)
  - **TabPFN v3.5** (`tabpfn` package 9.1.0, `create_default_for_version(V3_5)`, GPU): OOF 0.8262/0.8250/0.8243 (mean 0.8252). Fresh seeds gave 0.8231/0.8246. Mean +0.0078 vs CatBoost on the same features and folds. **Public LB 0.81786.** — [discoveries.md](https://raw.githubusercontent.com/TaeyanG4/kaggle-spaceship-titanic-experiments/main/discoveries.md); [README](https://github.com/TaeyanG4/kaggle-spaceship-titanic-experiments)
  - Relative to TabPFN v3.5: TabPFN v2.5 −0.0075 (i.e. roughly on par with CatBoost); TabICL −0.0183; TabDPT −0.0086 (OOF ~0.815-0.819); RealMLP −0.0382; TabM −0.0196; TabR −0.0313; TabSTAR −0.0160; xRFM −0.0292. A field-aware factorization machine scored OOF ~0.800-0.804. — [discoveries.md](https://raw.githubusercontent.com/TaeyanG4/kaggle-spaceship-titanic-experiments/main/discoveries.md)
  - TabPFN v3.5 fine-tuning: a single fine-tuned model gave +0.0020; an equal blend of frozen and fine-tuned gave +0.0017 (5/5 fresh seeds positive); a 32-estimator ensemble gave only +0.0002. — [discoveries.md](https://raw.githubusercontent.com/TaeyanG4/kaggle-spaceship-titanic-experiments/main/discoveries.md)
  - The project used 12 submissions and was 8th on the LB as of 2026-10-06. — [README](https://github.com/TaeyanG4/kaggle-spaceship-titanic-experiments)
- A second repo reports best public 0.82604 by reusing TaeyanG4's published TabPFN probabilities and retraining only a meta-model. This is derivative, not independent. — [GitHub: Ziiimu-H/spaceship-titanic](https://github.com/Ziiimu-H/spaceship-titanic)
- MLP (Keras-style, 3 hidden layers, dropout, ElasticNet L1/L2, Adam): score 0.80851, "within the top 200" (19 Aug 2023). It is not stated whether this is public LB; no CV is given. — [acciochris blog](https://acciochris.github.io/posts/spaceship-nn.html)
- Keras Sequential NN (2x128 hidden): max ~79% accuracy. — [petrosdemetrakopoulos/spaceship-titanic (via ecosyste.ms)](https://awesome.ecosyste.ms/projects/github.com%2Fpetrosdemetrakopoulos%2Fspaceship-titanic)
- TabNet notebook: public 0.79962 (search snippet; unverified). — [Kaggle: defcodeking TabNet](https://www.kaggle.com/code/defcodeking/tabnet-a-neural-network-for-tabular-data/notebook)
- TabPFN v2 (Nature, Jan 2025): reported to outperform "all previous methods on datasets with up to 10,000 samples by a wide margin". In classification, "in 2.8 s, TabPFN outperforms an ensemble of the strongest baselines tuned for 4 h." — [PubMed 39780007](https://pubmed.ncbi.nlm.nih.gov/39780007/); [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11711098/)
- TabPFN v2's hard limits are ≤10,000 training rows and ≤500 features. Spaceship Titanic (8,693 rows, ~24 features) fits within both. — [TabArena arXiv 2506.16791](https://arxiv.org/html/2506.16791v2)

### Inferences
- TabPFN v3.5 is the only model family with documented, seed-confirmed OOF gains over honest CatBoost on this exact competition with group-aware CV. Its gain (+0.008) is about 10x what we got from GBDT averaging (+0.0005). It is the top candidate to try. Caveats: the result comes from a single source, the CV-to-LB gap widened to about −0.007, and the claims are pending verification.
- Package version matters a lot. v2.5 was about 0.0075 worse than v3.5 in that study, so results with `tabpfn` v2 should not be expected to match. TabPFN v3.5 may need a GPU and may need license/API access; check before relying on it.
- Generic MLPs, TabNet and modern deep tabular nets (RealMLP, TabM) underperform here, unlike in their TabArena ranking.

### Gaps
- No FT-Transformer result specific to Spaceship Titanic was found.
- No Spaceship Titanic result for TabPFN v1 or v2 by itself (only v2.5 relative to v3.5) was found.
- No independent replication of the TabPFN v3.5 numbers was found.

## Q3. Stacking and blending with diverse base models

### Takeaway
Reported stacking gains on Spaceship Titanic are small (+0.001 to +0.013 over the best single model, depending on how weak the baseline is). In the most careful study, adding CatBoost or other diverse members to a TabPFN stack did *not* help. The meaningful stack gain (+0.003) came from combining different TabPFN variants with a nested logistic-regression meta-model.

### Cited Findings
- polarr/titan: a 10-model stack (RF, ExtraTrees, GB, XGB, LR, CatBoost, SGD, LDA, Bagging, MLP) with a HistGradientBoosting meta-model reports 0.81318 (rank 73/~2700), vs 0.80009 for the best single model (RF). It does not say whether the score is public or private; no dates; the baseline is weak (RF only). — [GitHub: polarr/titan](https://github.com/polarr/titan)
- AmirFARES: a weighted ensemble of the top 6 models scored 0.80 (top 28% of 2062). — [GitHub: AmirFARES](https://github.com/AmirFARES/Kaggle-Spaceship-Titanic)
- TaeyanG4: nested LR stacker on 3 TabPFN v3.5 members (frozen, fine-tuned, refit), with each fold's meta-model trained only on the other folds' OOF predictions. Nested OOF 0.8291/0.8288/0.8269, +0.0031 vs single frozen TabPFN (3/3 seeds), confirmed +0.0033/+0.0053 on fresh seeds. Public LB 0.82020. — [discoveries.md](https://raw.githubusercontent.com/TaeyanG4/kaggle-spaceship-titanic-experiments/main/discoveries.md)
- Same study, negative results: CatBoost into the TabPFN blend −0.0020; ten mixed members +0.0019 (below their +0.002 bar); non-linear meta-models (residual boosting −0.0024, segment LR −0.0004); CatBoost seed bagging +0.0006; supervised kNN features −0.0086; pseudo-labelling −0.0005. — [discoveries.md](https://raw.githubusercontent.com/TaeyanG4/kaggle-spaceship-titanic-experiments/main/discoveries.md)
- Same study: honest XGBoost was −0.0056 vs CatBoost; HistGB −0.0030; ExtraTrees −0.0116. LB order among 8 near-tied candidates did not follow CV order. — [discoveries.md](https://raw.githubusercontent.com/TaeyanG4/kaggle-spaceship-titanic-experiments/main/discoveries.md)
- TabArena: an ensemble of all TabArena models beats every individual model and AutoGluon. Post-hoc ensembling gains are large for LightGBM/RealMLP but small for CatBoost (Elo: CatBoost default 1432, tuned 1475, tuned+ensembled 1489, i.e. +57 total; LightGBM +326). — [TabArena arXiv 2506.16791](https://arxiv.org/html/2506.16791v2)

### Inferences
- Our observation (GBDT averaging gives +0.05 points because of 0.98-0.99 correlation) matches the literature. CatBoost gains the least from tuning and ensembling of any TabArena model. Stacking only helps with a genuinely different and *comparably strong* model; weak diverse models (LR, kNN, SVM, MLP at ~0.79-0.80) are unlikely to add accuracy on top of a 0.817 CatBoost.
- The large polarr/titan gain (+0.013) is relative to an RF baseline about 0.80, so it does not carry over to a strong CatBoost baseline.

### Gaps
- No Spaceship Titanic study was found that reports OOF for a diverse-family stack (LR/kNN/SVM/RF/GBDT/NN) on top of a strong CatBoost with group-aware CV.

## Q4. Threshold tuning, calibration and positive-rate matching

### Takeaway
On this balanced target, threshold tuning does basically nothing reliable: +0.0009 (2/3 seeds) on a CatBoost blend and −0.0023 on TabPFN in the only careful test. Keeping 0.5 is the documented choice. One blog mentions shifting the threshold to match train/test distributions, but its score is not clearly attributed to that technique.

### Cited Findings
- Nested threshold optimization on the CatBoost blend gave +0.000882 (positive on 2/3 seeds). On TabPFN it gave −0.002262. The 0.5 threshold was kept. — [discoveries.md](https://raw.githubusercontent.com/TaeyanG4/kaggle-spaceship-titanic-experiments/main/discoveries.md)
- Group-consistency post-processing (forcing travel-group members to agree) gave −0.0018 to −0.0023. — [discoveries.md](https://raw.githubusercontent.com/TaeyanG4/kaggle-spaceship-titanic-experiments/main/discoveries.md)
- A Japanese blog describes post-processing that shifts the threshold after comparing the train vs test prediction distributions (assuming identical distributions). It reports a quick GBDT at public 0.79307 (rank 1088/2238) and mentions 0.80874 without saying which model or method produced it. It also notes a "strange wall" in LB scores around 0.82. No date. — [scrapbox pokutuna](https://scrapbox.io/pokutuna/Spaceship_Titanic)
- CV-LB gaps from honest pipelines: MAMUT CatBoost CV 0.8094 → public 0.80617; LightGBM CV 0.8064 → public 0.79798; a household-disjoint CatBoost run had CV 0.8109 (group-bootstrap CI 0.8003-0.8214). — [MAMUT docs](https://mamut.readthedocs.io/en/latest/_sources/kaggle_benchmarks.rst.txt). TaeyanG4: gaps of −0.011 (leaky early stopping), −0.005 (honest CatBoost), and −0.007 (TabPFN). — [discoveries.md](https://raw.githubusercontent.com/TaeyanG4/kaggle-spaceship-titanic-experiments/main/discoveries.md)
- An honest re-implementation of the popular top public notebook (public 0.81833) had OOF about 0.804-0.806, roughly −0.012 vs honest CatBoost. That suggests its LB score is partly selection or noise. — [discoveries.md](https://raw.githubusercontent.com/TaeyanG4/kaggle-spaceship-titanic-experiments/main/discoveries.md)

### Inferences
- With a 50.4% positive rate and well-calibrated GBDT/TabPFN log-loss outputs, the optimal threshold sits very close to 0.5. Expected gains are within noise (±0.001), so this is not a lever for us.
- Our 0.8169 OOF vs 0.80266 LB gap (−0.014) is larger than the −0.005 achieved with honest inner-CV early stopping. It is worth checking whether our early stopping or tuning uses the scored fold. Fixing that improved LB by +0.0035 in TaeyanG4's study even though OOF fell.

### Gaps
- No Spaceship Titanic study of probability calibration (Platt or isotonic) affecting accuracy was found.
- No study of matching the test predicted positive rate to the training rate, with numbers, was found.

## Q5. General evidence: diverse ensembles vs tuned GBDTs on ~10k-row data

### Takeaway
Grinsztajn et al. (2022) found trees state-of-the-art on ~10k-row data. Newer benchmarks (TabArena 2025, TabPFN Nature 2025, TabPFN-2.5) show that by 2025 tabular foundation models dominate at small data sizes (≤10k rows), and that cross-family ensembles beat any single family. They also show CatBoost gains the least from tuning and ensembling. On our dataset size the expected lever is "add TabPFN", not "add more GBDTs or weak learners".

### Cited Findings
- Grinsztajn, Oyallon & Varoquaux (NeurIPS 2022 D&B, 45 datasets): tree-based models "remain state-of-the-art on medium-sized data (~10K samples) even without accounting for their superior speed." Tabular NNs need robustness to uninformative features, preservation of data orientation, and the ability to learn irregular functions. — [arXiv 2207.08815](https://arxiv.org/abs/2207.08815)
- TabArena (2025; 51 datasets, 500-250k rows): "with tuning and ensembling, the best deep learning methods are equal to or better than gradient-boosted decision trees." GBDT vs DL is a "false dichotomy" because both contribute to ensembles. "Tabular foundation models dominate for small data." TabPFNv2 "outperforms related approaches by a large margin" on its compatible subset, and tuned+ensembled TabPFNv2 beats AutoGluon there. CatBoost is #1 among default and tuned single models. — [arXiv 2506.16791](https://arxiv.org/html/2506.16791v2)
- TabArena Elo (tuned+ensembled): RealMLP 1569, TabM 1552, LightGBM 1532, CatBoost 1489, ModernNCA 1435, TabPFNv2 1414 (TabPFNv2 is penalized by RF imputation on datasets it cannot run). CIs are about ±20-30 Elo. — [arXiv 2506.16791](https://arxiv.org/html/2506.16791v2)
- TabPFN v2 (Nature 637:319-326, Jan 2025): beats a 4h-tuned ensemble of strongest baselines in 2.8 s for classification on ≤10k samples. — [PubMed](https://pubmed.ncbi.nlm.nih.gov/39780007/)
- TabPFN-2.5 (Nov 2025): "substantially outperform[s] tuned tree-based models" on TabArena (up to 100k rows) and matches AutoGluon 1.4 (4h tuned ensemble). — [arXiv 2511.08667](https://arxiv.org/pdf/2511.08667v2)
- An independent commentary (2025) notes TabPFN beating RF/XGBoost/LightGBM/CatBoost on 50+ benchmark datasets, while noting ensemble trees have been the gold standard. — [arXiv 2505.20003](https://arxiv.org/html/2505.20003v1)

### Inferences
- Benchmark gains are reported as Elo and average ranks, not accuracy points. On a noisy, balanced, near-ceiling task like Spaceship Titanic, the realized gain from switching families is small in absolute terms: about +0.008 OOF for TabPFN v3.5 vs CatBoost, the only measured case. Most other families (RealMLP, TabM) underperformed here despite strong TabArena ranks, so per-dataset variance is high.
- Realistic ceiling for honest pipelines: OOF about 0.825-0.829 and public LB about 0.818-0.826 (TabPFN-based). For CatBoost-based pipelines, about 0.817 OOF and 0.806-0.811 LB.

### Gaps
- TabArena per-dataset accuracy deltas for datasets similar to Spaceship Titanic (~9k rows, mixed categorical) were not extracted. Only aggregate Elo values were read.
- TabRepo was not covered for lack of tool budget.
