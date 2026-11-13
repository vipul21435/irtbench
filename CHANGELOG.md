# Changelog

## 0.1.0 (2026-11-13)

First version.

- Read long (CSV, TSV, JSON lines) and wide (CSV, TSV) results with missing cells.
- Rasch (1PL) and 2PL fits by marginal maximum likelihood with EM. The quadrature grid follows
  the width of the ability posteriors, so benchmarks with hundreds of items per model are
  fitted correctly; the ability distribution is re-standardised after every M-step.
- A normal prior on the ratio of each 2PL slope to a common slope (Bayes modal estimate),
  default standard deviation 0.5, which keeps the 2PL stable with few models.
- EAP abilities with posterior standard deviations; posterior predictive probabilities.
- Joint maximum likelihood with a ridge penalty, for comparison.
- Standard errors from the observed information and from a bootstrap over models.
- Item diagnostics: solved by all or by none, correlation with the rest score, negative
  discrimination, item and test information, Rasch infit and outfit.
- 1PL against 2PL: AIC, BIC, likelihood ratio, and held-out cells (log-likelihood, AUC).
- Subset selection (information, clustering, random), score prediction from a subset, and
  an evaluation that holds out models.
- Commands `fit`, `items`, `subset`, `predict`, `report`; charts with the `plot` extra.
- A fetch script and a study script for public SWE-bench Verified results.

Corrected before the first tag, after a review:

- The baseline of the subset evaluation without a model (`random_mean`, the plain mean of the
  answers on a random subset) now draws its items from every item of the data. It used to draw
  only among the items the fit estimated, which left out the items no model solves and made
  the baseline look worse than it is. On SWE-bench Verified its error at 25, 50 and 100 items
  was reported as 7.89, 5.74 and 4.62 points; drawn from all 500 items it is 7.19, 4.79 and
  3.24. The IRT estimator on a random subset (4.22, 2.95, 2.06) has about 60 % of that error,
  not about half. The README table, its text and the chart are from the new run.
- Long CSV and TSV input: a row with fewer fields than the header is an error that names the
  line, instead of a missing score or a model called "None". A byte-order mark at the start of
  an input file is skipped.
- `examples/answers.csv`, the file the quickstart's `predict` command reads, is in the
  repository, and a test runs the quickstart against it.
- The README says that the slope prior does not bound a common slope: on perfectly separated
  results the 1PL and the 2PL do not converge, and the fit reports it.
- The README no longer states how many chapters Birnbaum wrote, gives the study's timing with
  the conditions it was measured under, and says plainly that no third-party IRT package was
  compared against.
