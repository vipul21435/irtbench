# irtbench

Fit item response models to a models-by-items matrix of benchmark results, to find which items
separate models, which carry no information, and how small a subset of items predicts the full
score.

**Status: early scaffold.** Reading results files, the Rasch (1PL) and 2PL fits by marginal
maximum likelihood, ability estimates and the `irtbench fit` command exist and are tested.
Everything else in the plan is listed under "Not implemented yet", and two known problems of the
current fit are listed under "Limitations". Read both before relying on any number it prints.

## Why this is hard

A benchmark score counts every item the same. Some items are solved by every model and some by
none, so they say nothing about which model is better, and a few behave backwards: weaker models
solve them and stronger ones do not, which usually means the item is broken or mislabelled.
Telling these apart needs a model of how the chance of solving an item changes with the strength
of the model, fitted to a matrix that has few rows (models), many columns (items) and gaps.

## Install

```
pip install git+https://github.com/vipul21435/irtbench
```

Python 3.10 or newer. The only runtime dependency is numpy.

## Quickstart

The input is either long (one record per cell: `model`, `item`, `score` with score 0 or 1, as
CSV, TSV or JSON lines) or a wide CSV whose first column is the model name and whose other
columns are items. Absent or empty cells are missing.

`examples/toy_results.csv` is simulated data, not a real benchmark: 40 models on 24 items drawn
from a 2PL model with a fixed seed, with 27 cells removed (`python examples/make_toy.py`
regenerates it).

```
$ irtbench fit examples/toy_results.csv --irt-model 1pl --rows 5
# irtbench fit: toy_results.csv

- data: 40 models x 24 items, 27 of 960 cells missing
- model: 1pl by marginal maximum likelihood (EM, 61 Gauss-Hermite nodes)
- log-likelihood: -487.801 with 25 parameters; EM converged in 26 iterations
- items left out of the fit: 0 solved by all, 0 solved by none, 0 without data

## Items (hardest first, 5 of 24)

| item | status | solved | discrimination | difficulty |
|---|---|---:|---:|---:|
| item-10 | estimated | 0.03 | 0.87 | 4.60 |
| item-09 | estimated | 0.03 | 0.87 | 4.48 |
| item-04 | estimated | 0.10 | 0.87 | 2.81 |
| item-19 | estimated | 0.11 | 0.87 | 2.77 |
| item-11 | estimated | 0.16 | 0.87 | 2.22 |

## Models (strongest first, 5 of 40)

| model | answered | solved | ability | sd |
|---|---:|---:|---:|---:|
| model-32 | 24 | 0.71 | 1.42 | 0.52 |
| model-33 | 23 | 0.70 | 1.40 | 0.52 |
| model-36 | 22 | 0.68 | 1.35 | 0.53 |
| model-26 | 22 | 0.68 | 1.14 | 0.52 |
| model-05 | 22 | 0.64 | 1.08 | 0.53 |
```

`--json PATH` writes every item and model. `--rows 0` prints full tables. `--model-col`,
`--item-col` and `--score-col` rename the long-format columns.

From Python:

```python
from irtbench import estimate_abilities, fit_mml, read_matrix

data = read_matrix("examples/toy_results.csv")
fit = fit_mml(data, "1pl")
abilities = estimate_abilities(fit, data)
print(fit.converged, round(fit.log_likelihood, 3), fit.n_parameters)
print(data.items[0], round(float(fit.difficulty[0]), 2))
print(abilities.models[0], round(float(abilities.theta[0]), 2), round(float(abilities.sd[0]), 2))
```

```
True -487.801 25
item-01 -0.5
model-01 -0.76 0.52
```

## How it works

A model `i` with ability `theta_i` solves item `j` with probability

```
P(x_ij = 1 | theta_i) = sigmoid(a_j * theta_i + d_j),    theta_i ~ N(0, 1)
```

where `a_j` is the discrimination and `d_j` the intercept. The difficulty `b_j = -d_j / a_j` is
the ability at which the item is solved half the time. In the 1PL all items share one `a`; this
is the Rasch model with abilities of standard deviation `a`, written so that the 1PL is the 2PL
with equal slopes.

The abilities are integrated out with an `n`-point Gauss-Hermite rule and the item parameters
maximise the resulting marginal likelihood. EM alternates an E-step (each model's posterior
weight on every quadrature node, then per item the expected numbers of attempts and of solves at
each node) and an M-step (a weighted logistic regression per item, or with one shared slope for
the 1PL, by Newton steps with backtracking). It stops when no item parameter moves by more than
`1e-6` in an iteration.

An item whose observed cells are all 1 or all 0 has no finite maximum-likelihood intercept. Such
items are flagged `all_correct` or `all_incorrect` and left out of the fit. Missing cells are
skipped in the likelihood, which is only right if the reason a cell is missing does not depend on
what its score would have been.

The ability reported for a model is the mean of its posterior (EAP), with the posterior standard
deviation next to it.

## Limitations

- **Long benchmarks are fitted wrongly at present.** The quadrature grid is fixed. With hundreds
  of items per model, the posterior of an ability is narrower than the gap between nodes (0.40
  near zero for 61 nodes), and the estimates then depend on the grid. On simulated results of
  100 models on 500 items generated with a common slope of 1.5, the fitted 1PL slope is 1.07
  with 61 nodes and 1.36 with 300 nodes (rules much larger than that are rejected, because
  their outer weights underflow):

  ```python
  import numpy as np
  from irtbench import fit_mml, simulate_responses

  data, _ = simulate_responses(100, 500, kind="1pl", seed=5, discrimination=np.full(500, 1.5))
  for n in (61, 300):
      fit = fit_mml(data, "1pl", n_quad=n, max_iter=3000)
      print(n, round(float(fit.discrimination[0]), 2))
  ```

  Until the grid is refined to the width of the posteriors, use the fit on short tests only (the
  tests recover simulated parameters with 20 items).
- **The 2PL is unstable with few models.** On the 40-model example above,
  `irtbench fit examples/toy_results.csv` stops at the limit of 2000 EM iterations with the
  slopes of three items above 10 and still growing (two are above 100). Nothing regularises the
  slopes yet.
- Scores must be 0 or 1. One ability per model; no guessing parameter.

## Not implemented yet

Joint maximum likelihood, standard errors, item diagnostics (item-rest correlation, negative
discrimination flags, information curves, infit and outfit), 1PL against 2PL comparison and
held-out fit, subset selection and score prediction, the `items`, `subset`, `predict` and
`report` commands, charts, and a study on real public benchmark results. No real-data numbers
exist yet.

## Development

```
uv run --extra dev ruff check .
uv run --extra dev pytest -q --cov=irtbench
```

65 tests, 98 % line coverage on Python 3.12; the same tests pass on Python 3.10.

## Related tools

- [evalstats](https://github.com/vipul21435/evalstats): error bars and paired comparisons for
  evaluation scores. irtbench does not compute intervals or tests on scores.

## Licence

MIT. Copyright (c) 2026 Vipul Raj Jha.

Built by Vipul Raj Jha · https://vipul21435.github.io
