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
