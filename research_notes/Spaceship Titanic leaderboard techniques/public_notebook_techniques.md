# Public Notebook and Discussion Techniques for Spaceship Titanic (LB 0.81-0.84)

Research date: 2026-11-21. Kaggle notebook and discussion pages are JS-rendered; WebFetch returned only page titles for every kaggle.com/code and kaggle.com/discussion URL tried (eu1234, werenja, discussion/712563). Notebook contents below therefore come from GitHub READMEs, blog posts, and search snippets that describe them. Anything known only from a Kaggle page title is flagged.

Our baseline for comparison: CatBoost, 24 features, OOF 0.8169 (StratifiedGroupKFold on GroupId, 5 seeds), public LB 0.80266.

## Which public notebooks report LB >= 0.81, and what do they do differently?

### Takeaway
Every public, documented result at or above 0.815 that I could find in 2026 uses **TabPFN v3.5** (a tabular foundation model), alone or stacked, not a GBDT. Documented GBDT-only pipelines with feature engineering like ours land at 0.805-0.810 public LB. The top TabPFN result is 0.82604 (TaeyanG4, Oct 2026), reproduced independently by a second repo.

### Cited Findings
**TabPFN-based (the main thing we have not tried):**
- TaeyanG4, "Leak-free 2026 TabPFN Stack | Top Public 0.82604" (Kaggle notebook; GitHub write-up). Public LB progression on Oct 5-6, 2026: CatBoost baseline 0.80780 -> inner-CV early-stopping fix 0.81131 -> **TabPFN v3.5 alone 0.81786** -> nested stack of 3 TabPFN v3.5 variants 0.82020 (official champion) -> best of 8 CV-ranked candidates **0.82604**. Ranked 8th on 2026-10-06 with 12 submissions. — [TaeyanG4 GitHub](https://github.com/TaeyanG4/kaggle-spaceship-titanic-experiments)
- TaeyanG4 stack details: three TabPFN v3.5 variants (frozen, fine-tuned within folds, fine-tuned then refit), combined by logistic regression on logits trained only on other folds' OOF (nested). The frozen model gets a *negative* weight and the fine-tuned one a positive weight. The 0.82604 file adds a 100-epoch fine-tuned member. Exact TabPFN hyperparameters are not given. — [TaeyanG4 GitHub](https://github.com/TaeyanG4/kaggle-spaceship-titanic-experiments)
- TaeyanG4 CV: honest OOF 0.8291 / 0.8288 / 0.8269 on SGKF seeds 42 / 123 / 2026. TabPFN v3.5 beat a same-feature control by about +0.008 OOF on all three seeds. The nested stack added +0.0031. Fine-tuning alone added less than +0.002. — [TaeyanG4 GitHub](https://github.com/TaeyanG4/kaggle-spaceship-titanic-experiments)
- TaeyanG4 features: GroupId from PassengerId; **GroupSize and SurnameSize counted over train+test combined** (transductive, label-free); spend missing-value correction; CryoSleep rules; group and cabin-location features; surname categories; **KNN distance features**. — [TaeyanG4 GitHub](https://github.com/TaeyanG4/kaggle-spaceship-titanic-experiments)
- Models TaeyanG4 tested and dropped because they scored below TabPFN v3.5 and did not help the stack: XGBoost, LightGBM, ExtraTrees, TabPFN v2.5, TabICL, TabDPT, TabSTAR, RealMLP, TabM, TabR. The stacking gain came from fine-tuned variants of the same TabPFN, not from adding model families. — [TaeyanG4 GitHub](https://github.com/TaeyanG4/kaggle-spaceship-titanic-experiments)
- TaeyanG4 published its TabPFN member OOF and test probabilities as a Kaggle dataset (CC BY 4.0). — [Kaggle dataset, cited by Ziiimu-H](https://www.kaggle.com/datasets/taeyangg4/spaceship-titanic-tabpfn-member-predictions)
- Ziiimu-H (GitHub) independently reproduced **0.82604** by fitting a logistic-regression meta-model on the four published TabPFN probability sets. The same author's TabPFN + CatBoost blend scored 0.82534, which differs from the 0.82604 file on only 7 test rows. Their own CatBoost + FE pipeline peaked at 0.80593, and RF scored 0.80032. — [Ziiimu-H GitHub](https://github.com/Ziiimu-H/spaceship-titanic)
- kalidenaiarys, "spaceship-titanic-tabpfn-catboost-lb-0-82534" (Kaggle notebook; score known from the URL slug and the Ziiimu-H citation only, not opened). — [Kaggle notebook](https://www.kaggle.com/code/kalidenaiarys/spaceship-titanic-tabpfn-catboost-lb-0-82534)
- vltnkiz (GitHub, autonomous-agent project): adding **TabPFN-3.5 to CatBoost** gave CV +0.0072 to +0.0080, better in 15/15 folds, blend CV 0.82076, **public LB 0.81084 vs CatBoost-alone 0.80523**. TabPFN alone had CV 0.82104, higher than the blend. A CV stacker over TabPFN reached about 0.824 CV but was not adopted. TabM as a blend partner: -0.0012. — [vltnkiz issue #59](https://github.com/vltnkiz/kaggle-spaceship-titanic/issues/59)

**GBDT-only notebooks and write-ups (similar to ours):**
- malcolmlow (DEV, Sep 24, 2026): 5-fold CatBoost with domain features reached OOF 0.8163 and **public LB 0.80967** (#98). The jump came from deterministic imputation (surname->HomePlanet recovered 271 values; decks A/B/C/T->Europa and G->Earth; spend>0->CryoSleep False recovered 174; age<13 zero spend; age<18 not VIP) plus `Cabin_Region = Cabin_Num // 300`, Deck_Side, ZeroSpend, and grouped spend totals. Cabin_Region was among the top 5 CatBoost split features. The article gives no CatBoost hyperparameters beyond "ordered boosting". — [DEV](https://dev.to/malcolmlow/pocket-data-science-ii-reaching-the-kaggle-spaceship-titanic-top-6-on-android-with-antigravity-p4a)
- Same author: a 10-fold CatBoost+HistGB blend (70/30) at threshold 0.480 raised OOF to 0.8205 but **dropped LB to 0.80547**, with the test positive rate rising from 51.62% to 53.21%. A 10-fold tuned CatBoost at threshold 0.5 scored OOF 0.8172 and LB 0.80757, still below the simpler 5-fold model. — [DEV](https://dev.to/malcolmlow/pocket-data-science-ii-reaching-the-kaggle-spaceship-titanic-top-6-on-android-with-antigravity-p4a)
- eu1234, "Spaceship 81.1% Leaderboard Top 2% (Step by step)": self-reported best 0.81108 per a search snippet. Content could not be fetched; method unverified. — [Kaggle](https://www.kaggle.com/code/eu1234/spaceship-81-1-leaderboard-top-2-step-by-step)
- werenja, "Spaceship Titanic | Top 4% CatBoost + Feature Eng": title only. Cited as a source by Ziiimu-H. No score or method could be retrieved. — [Kaggle](https://www.kaggle.com/code/werenja/spaceship-titanic-top-4-catboost-feature-eng)
- surenvayradyan, "TOP REAL SCORE - LGBM + CatBoost": title only, no score retrieved. — [Kaggle](https://www.kaggle.com/code/surenvayradyan/top-real-score-lgbm-catboost)
- Other documented GBDT/NN scores, all below 0.81: XGBoost one-hot (max_depth 4, lr 0.1, 200 trees) 0.80243 ([zhongxiao37 blog, 2024](https://zhongxiao37.github.io/blog/2024/06/29/kaggle刷题之泰坦尼克号宇宙飞船/)); 3-layer NN with dropout and ElasticNet 0.80851, top 200 in Aug 2023 ([acciochris](https://acciochris.github.io/posts/spaceship-nn.html)); LGBM+CatBoost vote 0.80897 ([xup65k6t6 blog](https://xup65k6t6.github.io/Personal_Blog/projects/Kaggle:Spaceship-Titanic)); CatBoost 0.80687 ([Danial-Rashidi](https://github.com/Danial-Rashidi/spaceship-titanic-transport-prediction)); calibrated trio ensemble 0.80734 ([Reizen721](https://github.com/Reizen721/kaggle-competitions)); weighted ensemble 0.80, top 28% ([AmirFARES](https://github.com/AmirFARES/Kaggle-Spaceship-Titanic)).

### Inferences
- The one technique in the 0.815+ results that we have not tried is **TabPFN v3.5**, used alone, blended with CatBoost, or as a nested stack of fine-tuned variants. Two independent agent projects saw the same effect: roughly +0.007 to +0.008 CV over CatBoost and +0.006 to +0.010 public LB.
- Group/surname counts computed over train+test combined are a cheap transductive feature we may not have, if our group size uses train only. Worth checking.
- Our LB of 0.80266 is at the low end of the CatBoost results in comparable pipelines (0.8052-0.8097). With LB noise of about ±0.006, much of that gap may be noise. A large OOF-LB gap (0.0145 for us) is common: 0.8127->0.8052 for vltnkiz, 0.8188->0.8078 for TaeyanG4's CatBoost before its fix.

### Gaps
- I could not read any Kaggle notebook body, so I have no hyperparameters for eu1234, werenja or surenvayradyan.
- No TabPFN settings were published (n_estimators/ensemble size, fine-tune learning rate, epochs). The "100-epoch" detail is the only one.
- I found no public notebook in the 0.83-0.84 range with honest code. An MLE-bench/AgentBeats listing shows 0.83218 attributed to a Claude Sonnet 4.6 agent, known from a search snippet only and unverified: [AgentBeats](https://agentbeats.dev/agentbeater/mle-bench).

## Recurring tricks cited for this competition

### Takeaway
The commonly cited tricks are deterministic imputation, cabin-number regions, ZeroSpend/total spend, and group/surname sizes, and we have mostly done these already. Two cited items appear new for us: `Cabin_Num // 300` regions (if our features lack an explicit region bin) and KNN distance features. Several "tricks" are reported to *hurt*: threshold shifting, cabin-neighbour features, mate-outcome features, pseudo-labeling, and GBDT ensembling.

### Cited Findings
- **Cabin region binning:** `Cabin_Region = Cabin_Num // 300`, a top-5 CatBoost feature. — [DEV malcolmlow](https://dev.to/malcolmlow/pocket-data-science-ii-reaching-the-kaggle-spaceship-titanic-top-6-on-android-with-antigravity-p4a)
- **Surname features:** surname->HomePlanet lookup (no surname spans multiple planets) ([DEV](https://dev.to/malcolmlow/pocket-data-science-ii-reaching-the-kaggle-spaceship-titanic-top-6-on-android-with-antigravity-p4a)); SurnameSize and surname categories ([TaeyanG4](https://github.com/TaeyanG4/kaggle-spaceship-titanic-experiments)); a `no_surname` ablation switch ([Ziiimu-H](https://github.com/Ziiimu-H/spaceship-titanic)).
- **KNN distance features** are part of TaeyanG4's champion feature set. — [TaeyanG4](https://github.com/TaeyanG4/kaggle-spaceship-titanic-experiments)
- **Dropping GroupId as a raw feature:** vltnkiz's `impute+drop_group_id` was the best feature cell (+0.0013 CV, confirmed at only +0.0003), but it scored LB 0.80149 vs base 0.80523. — [vltnkiz #59](https://github.com/vltnkiz/kaggle-spaceship-titanic/issues/59)
- **Cabin-neighbourhood and mate-outcome (group-mate label) features hurt in every cell.** — [vltnkiz #59](https://github.com/vltnkiz/kaggle-spaceship-titanic/issues/59)
- **Threshold tuning hurts LB** (0.480 -> LB down 0.004). — [DEV](https://dev.to/malcolmlow/pocket-data-science-ii-reaching-the-kaggle-spaceship-titanic-top-6-on-android-with-antigravity-p4a)
- **One-hot vs native categoricals:** for XGBoost, one-hot gave 0.8258 holdout vs 0.8189 with native categoricals (single holdout, weak evidence). — [zhongxiao37](https://zhongxiao37.github.io/blog/2024/06/29/kaggle刷题之泰坦尼克号宇宙飞船/)
- **CatBoost depth:** depths 3/4/5 all scored below the default (-0.0064 / -0.0038 / -0.0005 CV). An earlier shallow-model LB of 0.80874 was judged noise. — [vltnkiz #59](https://github.com/vltnkiz/kaggle-spaceship-titanic/issues/59)
- **GBDT blending/stacking:** XGB, ExtraTrees and LR blended with CatBoost were all negative. Oracle analysis put combiner headroom over 5 GBDT-style models at about 0. — [vltnkiz #59](https://github.com/vltnkiz/kaggle-spaceship-titanic/issues/59). TaeyanG4 also found GBDT tuning, pseudo-labelling, label-noise removal and group post-processing did not help. — [TaeyanG4](https://github.com/TaeyanG4/kaggle-spaceship-titanic-experiments)
- **Early-stopping leak:** early stopping on the scoring fold inflated CV by about 0.004. Moving it to an inner split from the training fold raised LB from 0.80780 to 0.81131 (CatBoost family). — [TaeyanG4](https://github.com/TaeyanG4/kaggle-spaceship-titanic-experiments)
- **Group-blind vs row CV:** a gap of only 0.0008 ± 0.0004, with no change in idea rankings. — [vltnkiz #59](https://github.com/vltnkiz/kaggle-spaceship-titanic/issues/59). There is no GroupId overlap between train and test. — [DEV](https://dev.to/malcolmlow/pocket-data-science-ii-reaching-the-kaggle-spaceship-titanic-top-6-on-android-with-antigravity-p4a)

### Inferences
- The early-stopping finding is directly actionable. If our CatBoost uses `eval_set` = the validation fold with early stopping or use_best_model, our 0.8169 OOF is inflated by about 0.004, which explains part of our OOF-LB gap. The fix (inner split, or a fixed iteration count) also gave TaeyanG4 +0.0035 on LB.
- KNN-distance features and train+test group/surname counts are the only feature-level ideas in the top notebooks that are absent from our tried list.

### Gaps
- No source specified how the KNN distance features were built (which features, k, scaling).
- Age bins and spending ratios beyond LuxuryShare were not reported with measured effects in any source I could read.

## Highest LB scores with published code, and reproducibility

### Takeaway
The best honest, code-published score is 0.82604 (TabPFN v3.5 stack). It has been reproduced exactly from the published probabilities. Public scores above about 0.83, and especially the 0.89-0.965 at the top of the leaderboard, are not attributable to legitimate public code.

### Cited Findings
- 0.82604 was reproduced exactly by Ziiimu-H using TaeyanG4's published member probabilities and an LR meta-model. This reproduces the stacking step, not TabPFN training. — [Ziiimu-H](https://github.com/Ziiimu-H/spaceship-titanic)
- Even within TaeyanG4's own batch, LB order did not match CV order, and test-bagged versions of the same model differed by 0.0028 LB. — [TaeyanG4](https://github.com/TaeyanG4/kaggle-spaceship-titanic-experiments)
- The top 4 leaderboard teams scored 0.895-0.965, far above the best honest OOF of about 0.831. TaeyanG4 notes that some high-scoring public notebooks used external answer files or a "best public override" and excluded them. It also reports that a popular public notebook showing 0.81833 has an honest OOF of only about 0.805. — [TaeyanG4](https://github.com/TaeyanG4/kaggle-spaceship-titanic-experiments)
- TaeyanG4 states that the public LB is computed on the full test set, with no private split. This is a single source, unverified. — [TaeyanG4](https://github.com/TaeyanG4/kaggle-spaceship-titanic-experiments)
- The public test set has 4277 rows, which gives a binomial standard deviation of about ±0.006 on LB accuracy. — [vltnkiz #59](https://github.com/vltnkiz/kaggle-spaceship-titanic/issues/59)
- Top-100 threshold snapshot (Oct 2026): rank 100 = 0.80991. — [vltnkiz #59](https://github.com/vltnkiz/kaggle-spaceship-titanic/issues/59). 0.80967 was #98 in Sep 2026. — [DEV](https://dev.to/malcolmlow/pocket-data-science-ii-reaching-the-kaggle-spaceship-titanic-top-6-on-android-with-antigravity-p4a)
- A Kaggle discussion (per search snippet only; page not fetchable) reports one agent's 8 runs of the same code scoring 0.802-0.821. Two runs from the same Docker build, 14 minutes apart, scored 0.80460 and 0.80230. — [Kaggle discussion 712563](https://www.kaggle.com/competitions/spaceship-titanic/discussion/712563) (unverified)

### Inferences
- Public notebooks with high scores (≥0.818) and no visible TabPFN should be treated with suspicion, because TaeyanG4 found override or answer-file use. Do not chase their numbers.
- A single LB submission cannot resolve differences under about 0.01. Compare candidates on multi-seed group CV, as we already do.

### Gaps
- I could not verify the exact current top-public-notebook list or vote counts on Kaggle because the pages are JS-only.

## What discussion threads say about feature importance and the plateau

### Takeaway
With GBDTs, the practical ceiling on honest public LB is about 0.805-0.81. Several independent projects found feature and ensemble work exhausted at that level. The only documented break-through is switching model class to TabPFN v3.5, which reaches about 0.818-0.826.

### Cited Findings
- Permutation importance: raw columns dominate (CryoSleep and spend), followed by cabin, spend aggregates, and group. — [vltnkiz #59](https://github.com/vltnkiz/kaggle-spaceship-titanic/issues/59)
- vltnkiz concluded that model-layer and feature diversity are "largely spent". TabPFN was the only real gain. — [vltnkiz #59](https://github.com/vltnkiz/kaggle-spaceship-titanic/issues/59)
- Spending split: RoomService, Spa and VRDeck are negatively associated with Transported, and FoodCourt and ShoppingMall are weakly positive. — [DEV](https://dev.to/malcolmlow/pocket-data-science-ii-reaching-the-kaggle-spaceship-titanic-top-6-on-android-with-antigravity-p4a)
- The scrapbox notes by pokutuna mention Samuel Cortinhas's "Spaceship Titanic: A complete guide", a widely referenced notebook with group-based imputation by HomePlanet, Destination, Solo and Cabin_deck. No score was retrievable. — [scrapbox](https://scrapbox.io/pokutuna/Spaceship_Titanic); [Kaggle notebook](https://www.kaggle.com/code/samuelcortinhas/spaceship-titanic-a-complete-guide)

### Inferences
- The highest-expected-value next steps for us are: (1) TabPFN v3.5 alone and blended with our CatBoost under our existing SGKF harness; (2) checking for early-stopping leakage in our CatBoost CV; (3) a nested LR stack of TabPFN variants if TabPFN helps; (4) as a minor item, train+test GroupSize/SurnameSize and KNN distance features.
- Because of the LB noise of about ±0.006, our 0.80266 vs others' 0.805-0.810 GBDT results are not strong evidence that our features are worse.

### Gaps
- The Kaggle discussion forum itself could not be read. Discussion-level claims rely on GitHub/DEV write-ups, which are mostly from autonomous-agent projects in Sep-Oct 2026, plus one unverified snippet.
- TabPFN v3.5's licence, GPU needs and Kaggle-notebook availability were not researched.
