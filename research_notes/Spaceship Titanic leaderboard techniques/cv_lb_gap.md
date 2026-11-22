# CV-to-LB Gap in Spaceship Titanic: Causes, Diagnostics, Fixes

Our situation: CatBoost OOF accuracy 0.8169 (StratifiedGroupKFold by GroupId, 5 folds x 5 seeds, fold std ~0.012) vs public LB 0.80266. Gap = 1.42 points.

## Q1. Public vs private split, and the standard error of a public LB score

### Takeaway
Spaceship Titanic has no private leaderboard and no final deadline. Every one of the 326 distinct scores on the current leaderboard is exactly floor(k/4277) to 5 decimals, and no other denominator fits, so the public LB is scored on **100% of the 4,277 test rows** (inferred, but strongly). The binomial standard error of a score near 0.80 is about **0.61 accuracy points**. On its own, sampling noise accounts for roughly half the observed gap.

### Cited Findings
- Getting Started competition: "Final submissions selection is not relevant for Getting Started competitions." Start Feb 23, 2022, "This competition does not have an end date." 10 submissions/day. — Competition rules page, retrieved via Kaggle API (`kaggle competitions pages`) — [Rules](https://www.kaggle.com/competitions/spaceship-titanic/rules)
- There is a two-month rolling window. "Once a submission is more than two months old, it will be invalidated and no longer count towards the leaderboard. If your team has no submissions in the previous two months, the team will also drop from the leaderboard." — [Kaggle staff post "Rolling Leaderboards" (topic 309319)](https://www.kaggle.com/competitions/spaceship-titanic/discussion/309319); repeated in the competition FAQ ([Overview](https://www.kaggle.com/competitions/spaceship-titanic/overview))
- The same post says: "with a public ground truth and never-ending timeline, it was never the case that getting-started leaderboards were an honest representation of merit." A commenter (Philip AK H., 2023) notes that repeated submissions can extract information about the test labels. — [Topic 309319](https://www.kaggle.com/competitions/spaceship-titanic/discussion/309319)
- Test set: "remaining one-third (~4300) of the passengers"; metric is classification accuracy. — [Data / Evaluation pages](https://www.kaggle.com/competitions/spaceship-titanic/data)
- Kaggle's usual wording on split competitions is "This leaderboard is calculated with approximately 50% of the test data. The final results will be based on the other 50%". Other competitions use other percentages, e.g. ~96% for Allen AI and ~53% for a PyCon tutorial. I could not retrieve the exact banner text for Spaceship Titanic because the page renders with JS. — [DEV Community](https://dev.to/alan_scottencinas/everyone-knows-it-scores-half-nobody-checks-which-half-953); [Allen AI LB](https://kaggle.com/c/the-allen-ai-science-challenge/leaderboard/private); [PyCon 2015 LB](https://inclass.kaggle.com/c/pycon-2015-tutorial/leaderboard)
- **Own analysis of the downloaded public leaderboard** (`kaggle competitions leaderboard spaceship-titanic -d`, snapshot 2026-11-21 20:48 UTC, 2,135 teams):
  - I tested every denominator n from 1,000 to 4,277. Only **n = 4,277 with truncation** reproduces all 326 distinct scores (0 mismatches). The next best fits miss 167+ scores. Our 0.80266 = 3,433/4,277 = 0.802665, truncated.
  - Adjacent top scores differ by 0.00023-0.00024, which matches 1/4,277 = 0.000234.
  - Snapshot facts: the median team score is **0.80266**, exactly our score. Our score ranks about 1,048 of 2,135, and 22 teams are tied at it. Counts: 244 teams in [0.795, 0.800), 509 in [0.800, 0.805], 595 in (0.805, 0.810], 115 in (0.810, 0.815], 121 above 0.815. The top two scores (0.965, 0.931) are far above the rest (next is 0.839), which suggests label probing or leaked labels. An earlier snapshot the same day (17:56 UTC) had 2,161 rows, so the rolling window drops teams continuously.

### Standard error arithmetic (computed here)
- LB, n = 4,277, p = 0.80266: SE = sqrt(0.80266 x 0.19734 / 4277) = sqrt(3.703e-5) = **0.00609** (0.61 pt). 95% CI is about ±1.19 pt, i.e. 0.791-0.815.
- One test row = 1/4277 = 0.023 pt. One accuracy point = 42.8 rows.
- OOF CV, n = 8,693, p = 0.8169: SE = sqrt(0.8169 x 0.1831 / 8693) = sqrt(1.721e-5) = **0.00415** (0.41 pt). From folds instead: 0.012 / sqrt(5) = **0.0054**.
- SE of the difference, treating the two samples as independent: sqrt(3.703e-5 + 1.721e-5) = **0.00736**. Gap of 0.01424 / 0.00736 gives **z ≈ 1.93** (one-sided p ≈ 0.027). Using the fold-based CV SE: sqrt(3.703e-5 + 2.9e-5) = 0.0081, giving z ≈ 1.75.
- Clustering adjustment (own computation on train.csv): the one-way ANOVA intraclass correlation of the target within GroupId is 0.112. The size-weighted mean group size is 2.04 in train and 2.00 in test. That gives a design effect of about 1 + (2.0 - 1) x 0.11 ≈ 1.11, so the SEs above grow by about 5% (LB SE ≈ 0.0064). Caveat: this uses the ICC of the target, not of the model's errors, so it is an approximation.

### Inferences
- With 100% public scoring there is no public/private shake-up risk. The LB number is a single draw (SE about 0.6 pt) of the model's true accuracy on this population, and it is the final score.
- A 1.4-pt gap is about 2 SE of the difference. That is borderline: it could be pure noise plus modest CV optimism. It does not by itself prove a pipeline bug.
- Because the test labels are fixed and fully scored, differences between our own submissions are paired comparisons on the same 4,277 rows. The noise between two of our models is much smaller than 0.6 pt when they agree on most rows. It depends only on the rows where they disagree.

### Gaps
- I did not see the leaderboard banner text ("calculated with approximately X%") directly. The 100% figure is inferred from score granularity, but that inference fits exactly and nothing else does.

## Q2. What CV-vs-LB relationships do practitioners report?

### Takeaway
In the reports I found, LB is typically **lower** than CV by about 0.5-1.5 points: random-KFold CV around 0.81, group CV around 0.808-0.814, LB around 0.802-0.805. Run-to-run LB spread for nominally identical pipelines is about 1-2 points. Our 0.817/0.803 is at the high end of that gap but in the usual pattern.

### Cited Findings
- Basiru-Haruna (CatBoost, depth 6, 500 iters): random 5-fold CV **81.12%**, group-aware 5-fold CV **80.85%**, public LB **80.383%** (rank 788, Nov 2026). So gaps are 0.74 pt (random) and 0.47 pt (group). The author attributes part of it to "model-selection optimism" from reusing the same folds across experiments. The README wrongly mentions a private leaderboard. — [GitHub README](https://github.com/Basiru-Haruna/spaceship-titanic-ml)
- Georgy Mamarin (Kaggle discussion, Jun 2026): 8 runs of the same autonomous-agent setup scored **0.80230-0.82069** on the LB. Runs 7 and 8, from the same Docker build, scored 0.80460 and 0.80230. A deterministic rebuild (CatBoost/LightGBM/XGBoost stack) has **CV ~0.814**. He notes the MLE-bench gold bar is 0.82066 ("roughly top 6%"). — [Kaggle topic 712563](https://www.kaggle.com/competitions/spaceship-titanic/discussion/712563)
- oguntobore: StratifiedGroupKFold (5 folds) HistGradientBoosting **0.810 ± 0.016**, folds 0.780-0.827. The LB score is not reported. The author flags imputation on the full train set before CV as "a small source of optimism". — [GitHub](https://github.com/oguntobore/Spaceship-Titanic)
- xup65k6t6 blog: 85/15 holdout. The voting ensemble scored best on validation but worse on the LB than single LightGBM or CatBoost. LB **0.80897** (rank 133/2,504, Apr 2023). The author suspects leakage (unverified). — [Blog](https://xup65k6t6.github.io/Personal_Blog/projects/Kaggle:Spaceship-Titanic)
- A search-engine summary attributed a Towards AI write-up with "5-fold CV of 0.8563 with CatBoost, but a public score of 0.79665" and a note that the public score did not improve while CV did. **Unverified**: the page returned HTTP 403 when fetched. — [Towards AI](https://pub.towardsai.net/building-a-classification-model-to-score-80-accuracy-on-the-spaceship-titanic-kaggle-dataset-40a0aa184c1c)
- Amir Fares: a weighted ensemble of the 6 best models scored **0.80** on the LB (top 28% of 2,062). No CV reported. — [GitHub](https://github.com/AmirFARES/Kaggle-Spaceship-Titanic)

### Inferences
- A cluster of honest pipelines lands at LB 0.800-0.810 (509 + 595 teams on the current board) whatever the reported CV. The LB hardly changes as CV improves from 0.805 to 0.817. This is consistent with our plateau and with CV improvements of a few tenths being within noise or selection optimism.
- LB scores of 0.83 or more (about 121 teams above 0.815) probably include LB-probing or label-leak submissions, given the rolling LB, 10 submissions/day and fully public scoring. They are not a reliable target.

### Gaps
- I found no systematic survey or table of CV-vs-LB pairs for this competition, and no Kaggle Grandmaster write-up specific to it. Kaggle notebooks could not be scraped (JS-rendered).

## Q3. Does random KFold vs group-aware KFold systematically change CV here?

### Takeaway
Random or stratified KFold is slightly **optimistic** relative to group KFold, by about 0.3 pt in the one direct comparison. That is within fold-to-fold noise. Group leakage is real but small here, and it cannot explain a gap when you already use group folds.

### Cited Findings
- Random 5-fold 81.12% vs group-aware 80.85% (difference 0.27 pt) with the same CatBoost configuration. — [Basiru-Haruna](https://github.com/Basiru-Haruna/spaceship-titanic-ml)
- Four CV variants (StratifiedKFold vs GroupKFold, x group/deck aggregates computed on full data vs inside each fold) "all four land within the fold-to-fold spread, about a point wide, so the leak is real in principle but negligible here." — [Kaggle topic 712563](https://www.kaggle.com/competitions/spaceship-titanic/discussion/712563)
- Own computation: the target's ICC within GroupId is 0.112. Train has 6,217 groups and test has 3,063, with mean group sizes of about 2. So group membership carries a modest shared signal. That is consistent with a small leak under random KFold.

### Inferences
- The leak to watch in our pipeline is not the fold split itself. It is **transductive features**: group, cabin or surname aggregates, imputation from group-mates, or any target or group statistic computed on full train, or on train+test, before splitting. In CV the group-mate rows used for imputation or aggregation come from the same split. At test time, group-mates are all in test with no labels. Group-level target encodings and "group-mates' Transported" style features are the main suspects. If any feature uses train labels of group-mates, it is unavailable at test time, because no group spans train and test. That would give exactly a CV > LB gap.
- Pseudo-labeling (used in our round 3) can leak if pseudo-labels or test rows enter CV folds in a way that sees validation information. It is worth re-checking that CV numbers with pseudo-labels are computed with test pseudo-labels from models that never saw the validation fold. That is less important than the transductive-feature check above.

### Gaps
- No source quantifies the effect of group-mate target features specifically in this competition.

## Q4. Do threshold / class balance, full-train refit vs fold averaging, or seed averaging affect LB?

### Takeaway
I found no controlled, sourced reports for this competition. The available evidence is anecdotal, e.g. ensembles beating single models in CV but not on LB. The size of these effects can be reasoned about from the arithmetic: each 1-pt change is about 43 test rows.

### Cited Findings
- Ensembles or voting that win on validation can lose on LB. — [xup65k6t6 blog](https://xup65k6t6.github.io/Personal_Blog/projects/Kaggle:Spaceship-Titanic)
- Refit on all labelled data was used by the Basiru-Haruna and oguntobore pipelines. Neither reports fold-averaging as an alternative or states the threshold. — [Basiru-Haruna](https://github.com/Basiru-Haruna/spaceship-titanic-ml); [oguntobore](https://github.com/oguntobore/Spaceship-Titanic)
- Seed and run nondeterminism alone moved the LB by up to about 1.8 pt across 8 runs (agent-written code, so not purely seed). — [Topic 712563](https://www.kaggle.com/competitions/spaceship-titanic/discussion/712563)
- Adaptive reuse of a holdout or leaderboard causes overfitting to it. Blum & Hardt propose "the Ladder" to keep leaderboard estimates accurate under adaptive submissions. The abstract gives no numeric results. — [Blum & Hardt 2015, arXiv:1502.04585](https://arxiv.org/abs/1502.04585). The same logic applies to reusing the same CV folds for many experiments.

### Inferences (not sourced; derived from our numbers)
- **Class balance:** we predict 51.4% True vs 50.4% in train. The difference is 1.0 pt of the test set, about 43 rows. Even if every one of those rows were wrong, the threshold could cost at most about 1 pt, and realistically only a fraction of that, since rows near 0.5 are close to coin flips. It is unlikely to explain 1.4 pt. To check, compare OOF accuracy at thresholds 0.45-0.55 and at a "match train prevalence" cutoff. If OOF accuracy is flat within ±0.1 pt, leave the threshold at 0.5.
- **Full refit vs fold averaging:** the refit model trains on 25% more data and should be slightly better, not worse. But its iteration count and early-stopping point are not validated. If CV used early stopping per fold and the refit uses a fixed or averaged iteration count, behaviour can differ. Fold-model averaging (5 folds x 5 seeds = 25 models) is the version actually measured by OOF. Submitting it removes one source of mismatch. Compare test-prediction agreement between the refit and fold-average submissions. A disagreement rate of about 1-2% (43-86 rows) bounds the possible LB difference.
- **Seed averaging:** it reduces model variance and the randomness of the CV split, but **not** the sampling error of the 8,693 rows or the 4,277 test rows. The SE of 0.4-0.6 pt remains.
- **Selection optimism (likely the largest controllable piece):** 0.8169 is the best of many configurations (rounds 1-3: features, Optuna, ensembles, pseudo-labels) chosen on the same OOF folds. With fold std ~0.012, choosing the max of many correlated candidates can easily inflate CV by 0.3-0.8 pt. This is an estimate, not sourced. To fix, hold out a fixed nested or outer group fold that is never used for selection, or re-run the chosen pipeline on fresh fold seeds that were not used during tuning.

### Gaps
- No public A/B results for threshold tuning or refit vs fold-averaging on Spaceship Titanic were found.

## Q5. Any evidence the public test subset differs in distribution from train?

### Takeaway
No. The public subset is the whole test set, and our KS tests (p > 0.18) show no feature shift. I found no external reports of covariate shift. Group structure is nearly identical: mean group size 2.04 in train vs 2.00 in test.

### Cited Findings
- Train is ~two-thirds and test ~one-third of the passengers from the same ship manifest, with ID format `gggg_pp`. — [Data page](https://www.kaggle.com/competitions/spaceship-titanic/data)
- Own computation: size-weighted mean group size is 2.036 in train vs 1.997 in test. There are 6,217 train groups and 3,063 test groups.

### Inferences
- Because no group spans train and test, any feature that relies on labelled group-mates or on group statistics that include labels works in CV but is empty at test time. That is a "shift" in feature availability, not in raw covariates. KS tests on raw features will not detect it. To check, compute the distribution of each engineered group feature in OOF-validation rows vs test rows. Also compute the fraction of validation rows whose group-mates were partly in the training fold. With StratifiedGroupKFold that fraction should be 0, so if engineered features still differ, the leak is in feature construction.
- The test set's own base rate is unknown. If it is about 50%, a 51.4% predicted-True rate is fine.

### Gaps
- No adversarial-validation results for this competition were found in public sources. Running one (train-vs-test classifier on the engineered feature set, target AUC ≈ 0.5) is a cheap check worth doing.

---

### Diagnostic checklist (summary for the report writer; derived from the above)
1. Treat a 1.4-pt gap as about 1.9 SE. Noise could explain roughly half to all of it. Do not chase LB decimals: 1 row = 0.023 pt.
2. Re-estimate CV on fresh fold seeds or a never-touched outer group holdout to remove selection optimism.
3. Audit engineered features for full-train or train+test fitting (imputers, group/cabin/surname aggregates, target encodings, pseudo-labels). Recompute them inside each fold and compare CV.
4. Run adversarial validation on the engineered features, not the raw ones.
5. Submit the 25-model fold average instead of, or as well as, the full refit, and measure test-prediction disagreement between the two.
6. Check the OOF accuracy-vs-threshold curve. Keep 0.5 unless the gain is clearly larger than 0.2 pt.
7. Benchmark: the LB median is 0.80266. The honest-pipeline cluster is 0.800-0.810, and scores of 0.82+ are rare and often noise or probing.
