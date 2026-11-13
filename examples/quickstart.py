"""Fit, diagnose, choose a subset and predict, on the bundled simulated example.

    python examples/quickstart.py

Uses examples/toy_results.csv (simulated: 40 models, 24 items; see examples/make_toy.py).
"""

from pathlib import Path

import numpy as np

from irtbench import (
    diagnose_items,
    estimate_abilities,
    fit_mml,
    predict_scores,
    read_matrix,
    select_items,
)

data = read_matrix(Path(__file__).with_name("toy_results.csv"))
print(f"{data.n_models} models x {data.n_items} items, {data.n_missing} missing cells")

rasch = fit_mml(data, "1pl")
fit = fit_mml(data, "2pl")
print(f"1PL log-likelihood {rasch.log_likelihood:.2f}, 2PL {fit.log_likelihood:.2f}")

abilities = estimate_abilities(fit, data)
best = int(np.argmax(abilities.theta))
print(f"strongest model: {data.models[best]}, ability {abilities.theta[best]:.2f} "
      f"(sd {abilities.sd[best]:.2f})")

report = diagnose_items(data, fit, rasch=rasch)
weakest = int(np.nanargmin(report.discrimination))
print(f"lowest discrimination: {data.items[weakest]} at {report.discrimination[weakest]:.2f}, "
      f"rest correlation {report.rest_correlation[weakest]:.2f}")

chosen = select_items(fit, 8, "information", abilities=abilities.theta)
answers = data.take_items([data.items.index(item) for item in chosen])
predicted = predict_scores(fit, answers)
error = float(np.abs(predicted - data.model_means()).mean())
print(f"8 chosen items: {', '.join(chosen)}")
print(f"mean absolute error of the predicted full score (same models, in-sample): {error:.3f}")
