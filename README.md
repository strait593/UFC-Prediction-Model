# UFC Logistic Regression Modeling Snippet

I am pleased to demonstrate - UFC Prediction model. I have been working for a few days on the model itself and the API part for a about a week. This is a fun project however, it's not aimed at job recruiters,
job postings or anything that benefits me commercially, it's mine created by me for me for one of the things I love the most - mixed martial arts. Please do enjoy it tho :)

This is a small, runnable starting point for predicting whether the red or blue
corner wins a UFC bout. It uses the tracked
[`large_dataset.csv`](data/UFC-dataset/Large-set/large_dataset.csv) and the
existing pandas and scikit-learn dependencies.

## Run

From the repository root:

```bash
uv run python scripts/logistic_regression_bout_model.py
```

The script reports accuracy, balanced accuracy, ROC AUC (for a Red win), and a
classification report. A different dataset or holdout size can be selected with
`--dataset` and `--test-size`.

## Prediction app

Install/sync the project dependencies and start the FastAPI app:

```bash
uv sync
uv run uvicorn ufc_pred_model.api:app --reload
```

Open <http://127.0.0.1:8000> for the browser prototype. Search for fighters
from the included bout dataset, select Red and Blue corners, and optionally
choose a weight class. When omitted, the app chooses the division most often
shared in their recorded bouts. The JSON API is available at `/docs`:

- `GET /api/fighters?q=Justin` searches dataset fighter names.
- `GET /api/weight-classes?gender=Men` lists recorded divisions.
- `POST /api/predict` accepts `{"red_fighter":"Justin Gaethje","blue_fighter":"Ilia Topuria"}`.
- `GET /api/health` reports model availability and holdout metrics.

The app trains the existing logistic-regression model from the tracked large
bout dataset when it starts. Fighter statistics come from each fighter's first
appearance in the dataset's current row order. Probabilities and their implied
American/decimal odds are model estimates, not sportsbook lines; they are not
calibrated betting advice. The prototype currently only recognizes fighter
names present in that dataset.

## Assumptions and modeling choices

- `winner` is the target and only `Red` and `Blue` outcomes are retained.
- Features are intended to be available before the bout: pre-bout fighter
  record/statistic differences, weight class, gender, and each fighter's
  stance. Fight-result columns such as method, round, time, strikes, takedowns,
  and knockdowns are deliberately excluded to avoid target leakage.
- Missing numeric values are median-imputed; missing categorical values use the
  most frequent value. Numeric features are standardized and categoricals are
  one-hot encoded. Unknown categories at prediction time are ignored.
- The tracked large CSV has no bout-date column and is ordered newest to oldest.
  Its oldest 80% (the final rows) is used for training and its newest 20% (the
  first rows) is the holdout, with no shuffling. For production use, add a
  verified date column and sort by it before splitting.
- This is an initial baseline, not a calibrated betting or ranking system.
  Fighter-stat snapshots and the dataset's ordering should be independently
  verified before drawing conclusions from metrics.

## Sample prediction interface

The training script can score a JSON object with the same feature names:

```json
{
  "wins_total_diff": 2,
  "losses_total_diff": -1,
  "age_diff": -2,
  "height_diff": 3.0,
  "weight_diff": -1.5,
  "reach_diff": 2.0,
  "SLpM_total_diff": 0.8,
  "SApM_total_diff": -0.4,
  "sig_str_acc_total_diff": 0.03,
  "td_acc_total_diff": 0.05,
  "str_def_total_diff": 0.02,
  "td_def_total_diff": 0.10,
  "sub_avg_diff": 0.2,
  "td_avg_diff": 0.5,
  "weight_class": "Lightweight",
  "gender": "Men",
  "r_stance": "Orthodox",
  "b_stance": "Southpaw"
}
```

Save that as `bout.json`, then run:

```bash
uv run python scripts/logistic_regression_bout_model.py --predict-json bout.json
```

The output includes the predicted corner and probabilities for both corners.
