# Legitimacy of Top Spaceship Titanic Leaderboard Scores (0.93-1.00) and the Realistic Ceiling

Research date: 2026-11-21. Main sources: the Kaggle forum and competition pages, read through the official Kaggle CLI (`kaggle competitions topics/topic-messages/pages/leaderboard`), because kaggle.com discussion pages do not render through a plain web fetch. Kaggle forum links below use the form `https://www.kaggle.com/competitions/spaceship-titanic/discussion/<topic id>`. The CLI returned blank author names, so posters are identified only where the text itself names them.

## Q1: Is the dataset synthetic, who made it, and are the generating process or full labels public?

### Takeaway
The setting is plainly fictional (the year 2912, passengers "transported to an alternate dimension"), so the data must be artificially constructed. However, Kaggle has never published how it was generated, what Bayes error it was designed to have, or the test labels. Users asked staff about this in 2023 and 2024, and no staff answer was visible. I found no public source of full test labels.

### Cited Findings
- The competition is a Getting Started competition sponsored by Google LLC. It started on Feb 23, 2022 and has no end date. The API lists a placeholder deadline of 2030-01-01, which staff described as "a default deadline for all of our 'ongoing' competitions." — [Rules page (via Kaggle CLI)](https://www.kaggle.com/competitions/spaceship-titanic/rules); [Rolling Leaderboards thread #309319, staff reply 2022-03-04](https://www.kaggle.com/competitions/spaceship-titanic/discussion/309319)
- The Description page frames the data as fiction: "Welcome to the year 2912 ... collided with a spacetime anomaly ... almost half of the passengers were transported to an alternate dimension." — [Overview/Description](https://www.kaggle.com/competitions/spaceship-titanic/overview)
- The Data page says train.csv holds "about two-thirds (~8700) of the passengers" and test.csv "the remaining one-third (~4300)". It gives no information on how the data was generated. — [Data description](https://www.kaggle.com/competitions/spaceship-titanic/data)
- On 2023-05-18 a user wrote to "Ryan" (Ryan Holbrook, author of the welcome post) in the Welcome thread: "I would like to know more about how this dataset was created ... I'm assuming it was artificially generated." On 2024-07-07 another user asked: "Curious about the data generation process - was it made so that there's a specific ideal theoretical target accuracy achievable? Is it known?" I found no staff reply to either question among the thread's 282 messages that the CLI returned. — [Welcome thread #309323](https://www.kaggle.com/competitions/spaceship-titanic/discussion/309323)
- A search of Kaggle Datasets ("spaceship titanic full", "test answers", "labels", "ground truth", "solution", "original") found no dataset with test labels. "Spaceship Titanic completed" (lavagod, 2023) contains only train.csv, test.csv, train_target.csv and sample_submission.csv, and "Spaceship Titanic Solution" (nabarungos, 2022) contains only a Submission.csv. — Kaggle CLI `kaggle datasets list -s` / `kaggle datasets files`, run 2026-11-21
- A GitHub web search for leaked or full test labels found only mirrors of the competition CSVs and solution repos, with no labeled test file. — [AmirMAAI/Spaceship_Titanic_dataset](https://github.com/AmirMAAI/Spaceship_Titanic_dataset); [mziqudhd92/kaggle-spaceship-titanic-challenge](https://github.com/mziqudhd92/kaggle-spaceship-titanic-challenge)

### Inferences
- The original Titanic has a real external source of truth: public passenger and survivor lists. Spaceship Titanic is fictional, so no comparable external source exists. Anyone outside Kaggle could get the test labels only through (a) a leak of Kaggle's solution file, (b) recovering the generator, or (c) reading them back from leaderboard scores (probing). I found no evidence of (a) or (b).
- The noise you see (CryoSleep passengers on decks E and G only ~65% transported, awake non-spenders ~62%) fits a generator that adds stochastic noise to the labels. If so, part of the error is irreducible by design. This is unconfirmed, because Kaggle has not described the generator.

### Gaps
- No primary source from Kaggle (Addison Howard, Ashley Chow, Ryan Holbrook) describes the generation method. The search did not surface the competition's citation block naming those three authors, although the Kaggle page probably carries it.
- I could not confirm whether the 13,000 rows were derived from the original Titanic data (for example, synthesised from its structure) or generated from scratch.

## Q2: Forum threads on perfect/near-perfect scores, cheating, leaks, or probing, and what Kaggle staff say

### Takeaway
At least four forum threads (2023-2025) raise near-perfect scores: a 0.96 first place in May 2024, a 1.000 score in Aug 2024, and an explicit question about leaderboard probing in 2023. Community replies call the scores "leak or cheating", "fishy" or "not reachable". I found no Kaggle staff response in any of these threads. Since 2023, 0.93+ outliers have repeatedly appeared above a pack sitting at roughly 0.82-0.84.

### Cited Findings
- **Probing explicitly raised (2023-06-04):** a reply to Ashley Chow in the staff "Rolling Leaderboards" thread said: "if the leaderboard is not a good benchmark since you can draw a lot of (or even all of the) information by repeatedly submitting 'estimations' and evaluating the score that is returned … what could be a good benchmark instead for someone who is trying this challenge without 'cheating'?" No staff answer is visible in the thread's 9 comments. — [Rolling Leaderboards #309319](https://www.kaggle.com/competitions/spaceship-titanic/discussion/309319)
- **"Leaderboard 1st Place" (2024-05-02):** "the First Place scored something like 0.96. But the second place scored 0.82 ... is this big difference usual?" The only reply: "It is not typical, probably some leak or cheating." — [Thread #499696](https://www.kaggle.com/competitions/spaceship-titanic/discussion/499696)
- **"BAM! Someone scored 1.000!" (2024-08-24):** commenters said "There seems to be a lot of randomness with the target attribute, old 1st place was around 85% acc. Unless there's a hidden engineered feature which predicts target with 100% accuracy, which I sort of doubt". Others said "The same person also got 100% accuracy on another competition at the exact same time ... it's fishy" and "Rather sus" (2024-10-14). Nobody offered a method, and staff did not respond. (The thread names the account. Per the constraints I do not repeat it, and nothing in the thread establishes wrongdoing.) — [Thread #530046](https://www.kaggle.com/competitions/spaceship-titanic/discussion/530046)
- **"What is the best Algo ... above 95%?" (2025-01-30):** asked how they knew 95% was achievable, a user replied "cause leaderboard" (2025-08-22). Another user, on 2026-06-24, wrote: "that's not really reachable on this dataset. The public LB tops out around 0.83, and most strong solutions sit in the 0.80-0.82 range, so there's a hard ceiling from the noise in the data." — [Thread #560289](https://www.kaggle.com/competitions/spaceship-titanic/discussion/560289)
- **"Suspicious test accuracy of 1.000000" (2024-08-01)** turned out to be a beginner's target-leakage bug (Transported left in X), not a leaderboard result. — [Thread #523580](https://www.kaggle.com/competitions/spaceship-titanic/discussion/523580)
- **Independent practitioner (GitHub, Oct 2026):** "상위 4개 팀의 0.895–0.965는 이 프로젝트에서 검증한 어떤 모델의 정확도(정직한 OOF 최대 약 0.831)보다 훨씬 높다" ("the top 4 teams' 0.895-0.965 are far above any model validated in this project, honest OOF max ≈0.831"). The author explicitly declines to judge individual submissions ("개별 제출의 방법은 확인할 수 없으므로 판단하지 않는다"). The author also says they audited and excluded high-scoring public notebooks that rely on "external answer files, other people's submission CSVs, or hardcoded 'best public override' bits". This suggests that some public notebooks embed predictions copied from elsewhere instead of modeling. — [TaeyanG4/kaggle-spaceship-titanic-experiments](https://github.com/TaeyanG4/kaggle-spaceship-titanic-experiments)
- A GitHub web-search snippet attributed this claim to a repo author: public LB entries above 0.89 are "unreachable from the provided features" and fit a "test-label recovery or memorization pattern". I could not tie it to a specific page, so treat it as unverified. — search result referencing [TaeyanG4 repo](https://github.com/TaeyanG4/kaggle-spaceship-titanic-experiments)
- **Live leaderboard snapshot (Kaggle CLI, 2026-11-21):** #1 0.96539 (last submission 2026-11-05); #2 0.93149 (last submission 2026-11-21). Then a cliff: #3 0.83867, #4 0.83843, #5 0.83680, #6 0.83376, #7 0.83329, #8 0.83095, ranks 9-13 0.83072-0.83002, ranks 14-20 0.82908-0.82791. The competition list shows 2,134 teams. The rank-2 score was 0.92938 in an earlier search snapshot, 0.93126 in the brief's snapshot, and 0.93149 today. — `kaggle competitions leaderboard spaceship-titanic -s`; [Leaderboard](https://www.kaggle.com/competitions/spaceship-titanic/leaderboard)
- **Leaderboard mechanics relevant to probing:** a practitioner reports that the LB is "계산된다 ... test 데이터 전체로" (computed on the entire test set), so public score equals full-test accuracy and no private split exists. That matches another repo's statement that the public test has 4,277 rows. — [TaeyanG4 repo](https://github.com/TaeyanG4/kaggle-spaceship-titanic-experiments); [vltnkiz issue #59](https://github.com/vltnkiz/kaggle-spaceship-titanic/issues/59). The Evaluation page says only "classification accuracy". — [Evaluation](https://www.kaggle.com/competitions/spaceship-titanic/overview/evaluation)
- The limit is 10 submissions per day. — [Rules](https://www.kaggle.com/competitions/spaceship-titanic/rules)

### Inferences
- With scoring on the full test set and scores shown to 5 decimals, flipping one prediction moves accuracy by exactly ±1/4277 ≈ 0.00023, which is visible. Each submission can therefore reveal label information, and adaptive group-testing schemes can reveal more per submission. Starting from a ~0.83 model, reaching 0.965 means correcting about 0.135 × 4277 ≈ 580 rows. At 10 submissions/day that is feasible within weeks to months, even before using smarter probing. Leaderboard probing is therefore a plausible mechanism for 0.93-1.00. This is my inference, not something any source has demonstrated for a specific team.
- The rank-2 score has crept up in small steps (0.92938 → 0.93126 → 0.93149), which is what incremental probing would produce. Other explanations, such as blending in many small changes, cannot be ruled out. This is speculation and should not be presented as evidence against the team.
- Every 0.93+ score is 9-13 points above the dense 0.83 cluster. No public write-up explains any of them, and independent practitioners who audited their own validation top out at ~0.83 OOF. Together these strongly suggest the outliers do not reflect modeling skill. The specific mechanism (probing vs. copied answer files) is unverified.

### Gaps
- No Kaggle staff statement about cheating, probing, or the outlier scores in this competition was found. The CLI may not return nested replies, so a staff reply could have been missed.
- No team at 0.93+ has published a method.

## Q3: Getting Started rules and enforcement (external data, rolling leaderboard)

### Takeaway
The rules allow external data only if it is publicly and equally available. They forbid hand-labeling test data and multiple accounts. Enforcement is not visible: there are no prizes, the "final submission selection is not relevant," and staff have not responded publicly to the outlier threads. The 2-month rolling window invalidates old submissions, but a team that resubmits keeps its position, so the window does not purge probed scores.

### Cited Findings
- External Data clause (7.C): "You may use data other than the Competition Data ('External Data') ... However, you will ensure the External Data is publicly available and equally accessible to use by all participants". — [Rules](https://www.kaggle.com/competitions/spaceship-titanic/rules)
- "Submissions may not use or incorporate information from hand labeling or human prediction of the validation dataset or test data records." — [Rules](https://www.kaggle.com/competitions/spaceship-titanic/rules)
- "You cannot sign up to Kaggle from multiple accounts and therefore you cannot submit from multiple accounts." Also: "You may submit a maximum of 10 entries per day. Final submissions selection is not relevant for Getting Started competitions." — [Rules](https://www.kaggle.com/competitions/spaceship-titanic/rules)
- FAQ: "Getting Started competitions are a non-competitive way to get familiar with Kaggle's platform ... They have no cash prize and are on a rolling timeline." Also: "we have implemented a two month rolling window on submissions. Once a submission is more than two months old, it will be invalidated and no longer count towards the leaderboard. If your team has no submissions in the previous two months, the team will also drop from the leaderboard." — [FAQ on Overview page](https://www.kaggle.com/competitions/spaceship-titanic/overview/frequently-asked-questions)
- The staff post introducing the rolling window (2022-02-22) quotes @wcukierski: "Consider this a change to be more like a college class: your professor decides the curve based on how your classmates score, not based on all exam scores since 1986." — [Rolling Leaderboards #309319](https://www.kaggle.com/competitions/spaceship-titanic/discussion/309319)
- Users note that a top team "can submit the same code/ file once every 2 months and will retain the rank" (2024-08-24). — [Thread #530046](https://www.kaggle.com/competitions/spaceship-titanic/discussion/530046)

### Inferences
- Leaderboard probing is not named explicitly in the rules text I extracted. However, recovering test labels through submissions arguably conflicts with the spirit of the hand-labeling ban, and doing it at scale beyond 10/day would require multiple accounts, which is prohibited. Whether Kaggle takes action in Getting Started competitions is not publicly documented.
- Outlier scores have persisted (a 0.96 first place in May 2024, and 0.965 now). Either enforcement is not applied, or the same pattern recurs. Users should expect such scores to remain on the board.

### Gaps
- No public record of Kaggle removing scores from the Spaceship Titanic leaderboard was found.

## Q4: The realistic legitimate ceiling (and label noise / Bayes error)

### Takeaway
Legitimate results cluster at about 0.80-0.83 public LB. Carefully validated OOF accuracy tops out at about 0.83. The dense pack at ranks 3-20 (0.828-0.839) marks the practical frontier and is partly lucky: one LB reading has a binomial SD of about 0.006. Accuracy ≥ 0.81 already puts a team in roughly the top 100 (~top 5-6%). No formal Bayes-error analysis exists, but practitioners repeatedly report confidently wrong rows that do not respond to noise removal, which points to irreducible label noise.

### Cited Findings
- Leak-free experiments (Oct 2026): honest OOF maximum ≈0.831, with official-champion OOF of 0.8291/0.8288/0.8269 across seeds. Best public LB 0.82604 (best of 8 candidates), official champion 0.82020, CatBoost baseline 0.80780. "피처가 아니라 신호의 한계에 가까웠다" ("the limit was closer to the signal than to the features"). Label-noise removal, group post-processing and reimplementations of high-scoring public notebooks all failed to beat the promotion bar. Rows wrong in ≥4 of 5 seeds "were confidently wrong, with probabilities far from 0.5." LB order did not match CV order, and bagged variants of the same model differed by 0.0028 on LB. As of 2026-10-06 the author was in 8th place after 12 submissions, while the teams just above had submitted 81 and 82 times. — [TaeyanG4/kaggle-spaceship-titanic-experiments](https://github.com/TaeyanG4/kaggle-spaceship-titanic-experiments)
- Another documented effort (Sep-Oct 2026): rank 100 was 0.80991 on 2026-10-04 (97 teams ≥0.81 of 1,578). CatBoost had CV 0.81269; adding TabPFN raised CV to 0.82076 (TabPFN alone 0.82104) and public LB to 0.81084. "a single LB reading near 0.810 is partly luck", with binomial SD ≈0.006 on 4,277 rows. Feature headroom "looks spent on these inputs", and ensembling headroom was "~0". A goal of 0.85 CV was dropped. — [vltnkiz issue #59](https://github.com/vltnkiz/kaggle-spaceship-titanic/issues/59)
- Group-blind vs. group-aware CV gap was only 0.0008 ± 0.0004 (so group leakage in CV is small), and the residual correlation of group-mates was 0.043. — [vltnkiz issue #59, summarising tickets 69/97](https://github.com/vltnkiz/kaggle-spaceship-titanic/issues/59)
- Forum: "The public LB tops out around 0.83, and most strong solutions sit in the 0.80-0.82 range ... hard ceiling from the noise in the data" (2026-06-24). — [Thread #560289](https://www.kaggle.com/competitions/spaceship-titanic/discussion/560289)
- Forum, 2024-08-24: "There seems to be a lot of randomness with the target attribute, old 1st place was around 85% acc." This is an unverified recollection that conflicts with other sources' ~0.83 top; it may refer to a probed or outlier entry. — [Thread #530046](https://www.kaggle.com/competitions/spaceship-titanic/discussion/530046)
- Historical reference points: "top 10% (roughly 81% performance)" in Aug 2022 [Thread #346432](https://www.kaggle.com/competitions/spaceship-titanic/discussion/346432); "Top 10% | 80.734 Accuracy" in Dec 2023 [Thread #458708](https://www.kaggle.com/competitions/spaceship-titanic/discussion/458708); and the public notebook "Spaceship 81.1% Leaderboard Top 2%" [eu1234 notebook](https://www.kaggle.com/code/eu1234/spaceship-81-1-leaderboard-top-2-step-by-step). Percentile thresholds drift as the rolling board turns over.
- The live board on 2026-11-21 has ranks 3-20 at 0.82791-0.83867. — Kaggle CLI leaderboard

### Inferences
- A realistic legitimate ceiling is about 0.82-0.83 expected public LB, with single readings up to about 0.835-0.84 possible through selection luck (±0.006 SD; picking the best of many submissions inflates the maximum). Some of the 0.833-0.839 scores at ranks 3-7 probably include lucky selection or light LB-driven tuning, but nothing shows they are illegitimate.
- For your model, 0.8169 CV fits the legitimate frontier (~0.82-0.83 OOF). The 0.80266 LB is about 1.5 points below that CV and about 2-2.5 SD below the CV-implied expectation. This suggests that some LB headroom (to about 0.81-0.82) can be gained through variance or CV-LB alignment, not through new signal. Strong recent repos found TabPFN to be the one model addition that helped (+0.007-0.008 CV).
- Bayes error is probably about 17%, so roughly 0.83 is the maximum accuracy, judging from the consistent OOF ceiling and confidently wrong rows. This estimate is empirical, not formally derived.

### Gaps
- There is no formal Bayes-error or label-noise estimate for this dataset, and no Kaggle statement on the designed noise level.

## Q5: Is there an analogue to the original Titanic 1.0-score story?

### Takeaway
Partly. On the original Titanic, the 1.0 scores are widely attributed to the real passenger lists being public, plus leaderboard hacking. Spaceship Titanic has no public ground truth, so only the probing route (or a solution leak) applies. Community reactions are similar in both: "not possible without leakage." Kaggle staff are silent in both.

### Cited Findings
- On the original Titanic forum, commenters say 1.00 scores are not achievable without labeled test data or leakage, and note that the passenger data is public. — [Are the 1.0000 Titanic competition entries real?](https://www.kaggle.com/getting-started/458769); [How people are achieving 100% accuracy in Titanic ML competition?](https://www.kaggle.com/discussions/general/214175)
- A tutorial says the full Titanic passenger and survivor lists are public and that some leaderboard entries simply copy the true answers. — [Neurohive Kaggle Titanic tutorial](https://neurohive.io/en/guides/kaggle-titanic-tutorial/)
- Perfect Titanic scores are attributed to "leaderboard hacking", meaning iteratively changing single predictions until reaching 100%. — search-result summary of [Towards Data Science, "Introduction to Kaggle and Scoring Top 7% in the Titanic Competition"](https://towardsdatascience.com/introduction-to-kaggle-and-scoring-top-7-in-the-titanic-competition-7a29ce9c24ae/) (not fetched directly)
- A GitHub project calls 100% Titanic submissions illegitimate and cites 0.84668 (Chris Deotte, WCG+XGBoost) as the highest legitimate Titanic score. — [Rob-Bruton/Kaggle-Titanic](https://github.com/Rob-Bruton/Kaggle-Titanic) (via search snippet)
- The Titanic labels trace back to Encyclopedia Titanica (compiled by Frank Harrell and updated by Thomas Cason), and extended datasets merging Wikipedia and Encyclopedia Titanica exist on Kaggle. — [atmarkit article](https://atmarkit.itmedia.co.jp/ait/articles/2007/02/news016.html); [pavlofesenko titanic extended](https://baselight.app/u/kaggle/dataset/pavlofesenko_titanic_extended)

### Inferences
- The Titanic's external-data shortcut is not available here. For Spaceship Titanic, the only well-supported route to 0.93+ is extracting label information from the leaderboard itself, or copying another party's recovered predictions. A one-flip-at-a-time probing pattern fits the observed incremental score rises, but it remains unproven for any individual team.

### Gaps
- No Spaceship-Titanic-specific write-up was found that demonstrates or documents a probing attack, or that identifies a leaked solution file.
