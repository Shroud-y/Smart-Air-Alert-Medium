# SAAM — air raid alert forecast for Kyiv

A contest MVP: a console program that shows a map of current air raid alerts in Ukraine and forecasts, for the **city of Kyiv**:

- if there is no alert — **whether one will start** within the next `H` minutes;
- if an alert is active — **whether it will end** within the next `H` minutes.

Default `H` = 30 min. The user-facing output is binary (yes / no); probabilities and technical details live in dashboards.

> Status: plan and stack agreed, data verified. No model code yet.
> This file is extended as the work goes on: decisions, data analysis, changes of mind, corrections in both directions.

## Why Kyiv only

The chance of an alert in Kyiv depends strongly on the state of neighbouring oblasts (Chernihiv, Kyiv oblast and others). That gives the model a strong signal and keeps the task feasible in two days. The hypothesis is supported by [arXiv 2411.14625](https://arxiv.org/html/2411.14625v1): the state of neighbouring oblasts is the most important group of features.

## Key decisions

| # | Decision | Why |
|---|---|---|
| 1 | Two separate tasks: alert "start" and alert "end" | A combined target "alert active in N min" is explained almost entirely by an alert simply continuing, which gives misleadingly high metrics |
| 2 | Models: a probabilistic algorithm **and** LightGBM, both compared against baselines | A neural network was rejected: Kyiv has only ~2.3k alert starts, attack patterns drift, and in 2 days it is unlikely to beat boosting |
| 3 | Historical data: `official_data_uk.csv` from [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) (MIT) | Official source, starts 2022-03-15, Kyiv city is a separate region. Volunteer data rejected as riskier |
| 4 | Aggregation to oblast: an oblast is under alert if **at least one** of its raions or hromadas is | The data mixes levels (oblast / raion / hromada), raion-level dominates from 2025. The rule gives one representation for all years |
| 5 | Alerts separated by a short pause are **not merged** | Merging would distort the data and the "start" target |
| 6 | Exact duplicate rows are removed by the loading code; the raw CSV in git is not modified | Reproducibility, and the raw file can always be checked against the original |
| 7 | Time: storage and all interval computations in **UTC**; hour of day and day of week in **Kyiv time** (`Europe/Kyiv`) | See "Time" |
| 8 | Live alerts: [ubilling.net.ua/aerialalerts](https://ubilling.net.ua/aerialalerts/) in `?raw` mode | No token; returns raions, start time (UTC) and alert level. alerts.in.ua rejected: token by application, may not arrive in time |
| 9 | `--replay <datetime>` mode: plays back history from the dataset | Fallback if the API is unavailable, and a way to demo the program on a real attack |
| 10 | Interface: console, tile map of Ukraine (one cell per oblast), forecast below. No web app | A true ASCII outline is expensive and fragile. Rendering is one function; if tiles do not work out, it is replaced with a list of oblasts |
| 11 | Dashboards: matplotlib figures in `reports/figures/`, embedded in this README; a `--debug` panel in the console | Needed for debugging and for describing the data analysis |
| 12 | Tests with pytest | The main risk is leaking the future into features; it gives falsely good metrics and is invisible to the eye |

## Data

`official_data_uk.csv` is a snapshot; the last record is 2026-09-07 (the official file in the source repository has not been updated since).

- Columns: `oblast, raion, hromada, level, started_at, finished_at, source`.
- 299,275 rows, of which **185,430** are unique. 113,845 rows appear exactly twice.
- Duplicates are exact copies: all 7 columns match. Rows with the same place and start but a different end or level: 0. Duplicates exist up to 2026-01-25, none after.
- Empty `finished_at`: 0.
- Every timestamp carries the `+00:00` suffix.

To check the row and unique counts (should print `299275 185430`):

```bash
python -X utf8 -c "import csv;r=list(csv.reader(open('official_data_uk.csv',encoding='utf-8')))[1:];print(len(r),len(set(map(tuple,r))))"
```

### Kyiv after removing duplicates

| Year | Rows in file | Alerts | Hours under alert |
|---|---|---|---|
| 2022 (from 15 Mar) | 1,010 | 505 | 541 |
| 2023 | 602 | 301 | 397 |
| 2024 | 1,004 | 502 | 542 |
| 2025 | 1,022 | 511 | 740 |
| 2026 (to 7 Sep) | 503 | 448 | 405 |
| **Total** | 4,141 | **2,267** | |

Duration: median 38 min, 95th percentile 231 min, maximum 673 min. No overlapping intervals. 10 alerts are shorter than 2 min; 29 pairs of alerts are less than 5 min apart (not merged, decision 5).

## Time

The dataset is in UTC with a constant offset all year. Checked against known events: the mass strike of 2022-10-10 starts at 03:47 in the data, which is 06:47 Kyiv summer time.

The problem: Ukraine switches between UTC+2 (winter) and UTC+3 (summer), so the same UTC hour is a different hour for people. The data shows it: the peak of alert starts in Kyiv moves from 21:00 UTC in summer to 22:00–23:00 UTC in winter, while in Kyiv time it stays around midnight (2024–2026, small sample).

Options considered:

| Option | Idea | Verdict |
|---|---|---|
| **A** | Computations in UTC, calendar features in `Europe/Kyiv` | **chosen** |
| B | Convert the whole dataset to Kyiv time | The autumn hour 03:00–04:00 occurs twice, the spring one does not exist; an offset would have to be stored next to every timestamp |
| C | Everything in UTC + a "summer time" feature | Baselines cannot learn the shift, so the model comparison becomes unfair |
| D | Fixed offset all year | The hour is off by one for half of the year |

How A works: the time grid, region states, durations, lags and the target are computed in UTC, where a day is always 24 hours. For each grid point the Kyiv hour and day of week are computed separately with `zoneinfo`. On a switch night the grid stays regular; in autumn two neighbouring hours simply get the same label, and in spring one label is skipped.

No Kyiv alert in the dataset spans a switch moment (checked on all 9 switches); other oblasts were not checked.

## Evaluation

**Metrics**

- PR-AUC — primary (classes are imbalanced).
- Brier score and calibration curve.
- Precision / recall / F1 at a threshold picked on validation; the yes / no answer comes from this threshold.
- Number of false "yes" per day.

**Baselines**

1. Climatology: frequency of an alert start by Kyiv hour × day of week.
2. Neighbour rule: "yes" if Kyiv oblast or an adjacent oblast is under alert right now.
3. Logistic regression on the current state of oblasts only.

For the "end" task: the empirical duration curve — P(end within `H` min | alert has lasted `t` min).

**Time split** (no shuffling, gap ≥ `H` between parts)

| Part | Period |
|---|---|
| Train | 2022-03-15 … 2026-03-31 |
| Validation | 2026-04-01 … 2026-06-30 |
| Test | 2026-07-01 … 2026-09-06 |

The dates may be adjusted after EDA (old years may hurt because attack patterns changed); any change will be recorded here.

## Known limitations

- **Data gap.** History ends on 2026-09-07; there is more than a month without data between training and live use.
- **Yellow and red alerts.** Since September 2026 alerts carry levels (yellow — drones, red — missiles); ubilling returns them. The model is trained on data without levels, so the live forecast is outside the training distribution and there is nothing to measure its quality on under the new regime. The introduction date (6 September) comes from a third-party project, not an official source.
- **Change of granularity.** After the move to raion-level alerts (2025) the "oblast under alert" feature may fire more often than before. To be checked in EDA.
- **The live source is volunteer-run.** Ubilling proxies third-party sources that do not always agree with each other; not for safety decisions.

## Corrections log

Who corrected whom and what changed.

| Who | Before | After |
|---|---|---|
| Claude → author | The choice was "probabilistic algorithm or neural network" | A middle option was added — gradient boosting; the neural network was rejected |
| Claude → author | If the API is unavailable, forecast from day and time only, without the current state | A token-free API (ubilling) was found; the current state of neighbours stays the main signal |
| Author → Claude | Claude wrote that the dataset is updated daily | Only the volunteer data is updated daily; the official file stopped on 2026-09-07. The test period was shortened to 2026-09-06 |
| Author → Claude | Claude suggested not committing the CSV | The CSV is committed to the repository |
| Claude → Claude | "Before December 2025 alerts were oblast-level" | Hromada-level rows exist from 2022, the move to raions happened during 2025 |
| Author, verified by Claude | Suspicion: 1,010 alerts in Kyiv in 2022 is too many | Exact duplicates found across the whole file; the real count is 505 |
| Author, verified by Claude | Suspicion: time in the dataset is UTC and summer time will cause a problem | Confirmed; option A chosen |
| Claude → Claude | Argument against option B: alerts spanning a switch would get a wrong duration | No Kyiv alert spans a switch; only the code-complexity argument remains |
| Author → Claude | Claude wrote the README in Ukrainian | The whole project is in English |

## Stack

Python 3.11 · pandas · numpy · scikit-learn · lightgbm · matplotlib · rich · requests · pytest

## Plan

1. EDA: oblast states over time, the link between neighbours and Kyiv, the effect of the move to raion-level alerts → figures in `reports/figures/`.
2. Data loading and cleaning, time grid, features, target + leak tests.
3. Baselines and metrics.
4. Probabilistic algorithm and LightGBM, comparison.
5. Ubilling client, `--replay` mode.
6. Console interface: map, forecast, `--debug` panel.
