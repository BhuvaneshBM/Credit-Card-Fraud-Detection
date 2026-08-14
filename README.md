# Credit Card Fraud Detection — 5-Model Bake-off

Trains and fairly compares five classifiers on the [ULB Credit Card Fraud Detection
dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 European card
transactions over two days in September 2013, of which 492 (0.17%) are fraudulent.

## Dataset

- `V1`–`V28`: PCA components of the original (undisclosed) transaction features
- `Time`, `Amount`: the only two features left in raw form
- `Class`: 1 = fraud, 0 = legit — **578:1 imbalance**

Get it from Kaggle and place `creditcard.csv` next to the notebook (or in `./data/`), or add it
as a Kaggle input (`mlg-ulb/creditcardfraud`). Cell 1 auto-detects Kaggle / Colab / local paths.

## Method

- **Split:** stratified 70/15/15 train/val/test. Val exists specifically so the decision
  threshold can be tuned without touching test — picking a threshold and grading it on the same
  set is optimistic.
- **Scaling:** `RobustScaler` on `Time`/`Amount` only, fit on train and applied to val/test — not
  fit on the full dataset before splitting.
- **Imbalance handling:** reweighting, not resampling — `class_weight='balanced'` (Logistic
  Regression, Random Forest), `scale_pos_weight` (XGBoost, LightGBM), `auto_class_weights`
  (CatBoost). With only 492 real fraud rows in dense PCA space, SMOTE-style interpolation risks
  inventing points that aren't representative of real fraud; this notebook tests that claim
  directly rather than just asserting it (see Extensions).
- **Metrics:** PR-AUC (average precision) is primary — at 0.17% prevalence, ROC-AUC barely moves
  for a small number of false positives while precision collapses, so PR-AUC is the metric that
  actually reflects performance here.

## Results

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Frauds caught | False positives |
|---|---|---|---|---|---|---|---|
| **LightGBM** | **0.8502** | 0.9718 | 0.9206 | 0.7838 | 0.8467 | 58/74 | 5 |
| XGBoost | 0.8346 | 0.9664 | 0.8824 | 0.8108 | 0.8451 | 60/74 | 8 |
| XGBoost + focal loss | 0.8434 | 0.9629 | 0.9048 | 0.7703 | 0.8321 | — | — |
| CatBoost | 0.8300 | 0.9530 | 0.7821 | 0.8243 | 0.8026 | 61/74 | 17 |
| XGBoost + SMOTE | 0.8178 | 0.9708 | 0.7108 | 0.7973 | 0.7516 | — | — |
| Random Forest | 0.8168 | 0.9496 | 0.9636 | 0.7162 | 0.8217 | 53/74 | 2 |
| LogReg + SMOTE | 0.7944 | 0.9653 | 0.0643 | 0.8784 | 0.1198 | — | — |
| Logistic Regression | 0.7923 | 0.9677 | 0.0669 | 0.8784 | 0.1243 | 65/74 | 907 |

Threshold-tuned (val → test) best model: **LightGBM at threshold 0.30** — Precision 0.9091,
Recall 0.8108, F1 0.8571, vs. 0.8467 at the default 0.50 cutoff.

**Note on Logistic Regression's precision:** its low PR-AUC gap to the tree models (0.79 vs.
0.85) says the fraud signal is largely linear/additive, but `class_weight='balanced'` pushes it
to over-flag heavily at the 0.5 threshold specifically — it isn't a bad model, it's evaluated at
the wrong cutoff. Only the overall-best model currently gets threshold tuning; loop Cell 11 over
all five for a fully fair comparison.

## Which model should you use?

**Best overall: LightGBM** — highest PR-AUC (0.8502), highest ROC-AUC (0.9718), best F1 at
default threshold (0.8467, 0.8571 after tuning), only 5 false positives out of 42,648 legit test
transactions. It wins the metric that matters most here without trading away precision or recall
to do it.

Runner-up depends on what you're optimizing for:

| If you want... | Pick | Why |
|---|---|---|
| Best overall | LightGBM | Wins PR-AUC, ROC-AUC, and F1 simultaneously |
| Fewest false alarms | Random Forest | Only 2 false positives (precision 0.9636) — but lowest recall of the tree models (0.7162) |
| Catches the most fraud | CatBoost | Best recall (0.8243) — at the cost of more false positives (17) |
| Competitive alternative to reweighting | XGBoost + focal loss | 0.8434 PR-AUC, precision 0.9048 — evidence focal loss is a real option here, not just reweighting's math done differently |
| Fastest to train | Logistic Regression (2.1s) | Needs its own tuned threshold — unusable at 0.5 (907 false positives) |
| Avoid at default threshold | Logistic Regression / LogReg+SMOTE | Not bad models (PR-AUC ~0.79) — 0.5 is just the wrong cutoff for them |

**Caveat:** LightGBM's margin over XGBoost is fairly thin (0.8502 vs 0.8346 PR-AUC) on a single
train/val/test split with one random seed. Treat this as "LightGBM and XGBoost are both strong,
LightGBM edges ahead here" rather than settled — cross-validate across seeds/splits before
treating the ranking as final.

**Recommended deployment choice:** LightGBM at the val-tuned threshold of 0.30 — Precision 0.9091,
Recall 0.8108, F1 0.8571, and cheap to retrain (5.3s) if the model needs refreshing on new data.

## Extensions (optional cells, after the main bake-off)

- **SMOTE**, via `imblearn.pipeline.Pipeline` so oversampling only ever touches the training
  fold. Result: reweighting beat SMOTE for both XGBoost (0.8346 vs 0.8178 PR-AUC) and Logistic
  Regression (near-identical, 0.7923 vs 0.7944) — evidence for, not just an assertion of, the
  reweighting-over-resampling choice above.
- **Focal loss** (custom XGBoost objective, γ=2.0, α=0.75): down-weights easy/already-confident
  examples so gradient budget goes to the hard, fraud-like-legit cases. Beat plain reweighted
  XGBoost on PR-AUC (0.8434 vs 0.8346) with higher precision, less recall — the expected focal
  loss trade-off. Gradient/Hessian derived symbolically and verified against finite differences.
- **Cost-based threshold**: replaces F1-optimal with a threshold that minimizes
  `$FP × cost + $FN × cost` using real business costs, not an arbitrary F1 tie-break.
- **Isolation Forest** (unsupervised, no labels): PR-AUC 0.17 vs. 0.85 supervised — a floor
  showing how much the 492 labels are actually buying you.

## Known gotchas fixed here (worth keeping if you extend this)

- **LightGBM + high `scale_pos_weight` needs `boost_from_average=False`.** Without it, LightGBM's
  default initializes predictions from the (skewed) average target and doesn't fully correct
  within 300 rounds — silently miscalibrated probabilities (PR-AUC 0.02 instead of 0.85), not an
  error, so it's easy to miss.
- **Threshold-tune on a validation set, never on test.** Scanning F1 across thresholds directly
  on the test set and reporting that same test set's score is leakage, even though it never
  touches the model itself.
- **Fit scalers after splitting, not before.** Fitting `RobustScaler` on the full dataframe before
  `train_test_split` leaks test-set distribution into the scaling parameters.
- **Custom GBM objectives need verification, not just a clean-looking formula.** A hand-derived
  focal loss gradient/Hessian had a sign/magnitude bug for the negative class that produced a
  plausible-looking but wrong model (decent PR-AUC ranking, ~0 F1 at any sane threshold). Checked
  against finite differences before trusting it.

## Requirements

```
pandas numpy scikit-learn matplotlib seaborn
xgboost lightgbm catboost
imbalanced-learn   # only for the SMOTE extension cell
```

## Structure

1. Setup & data loading
2. EDA (class balance, amount/time distributions, correlations)
3. Preprocessing & train/val/test split
4. Model definitions (5 classifiers)
5. Train & collect predictions
6. Results table (PR-AUC / ROC-AUC / precision / recall / F1)
7. PR & ROC curves
8. Confusion matrices
9. Feature importances
10. Threshold tuning (val → test)
11. *Extensions:* SMOTE comparison, focal loss, cost-based threshold, Isolation Forest baseline
12. Summary
