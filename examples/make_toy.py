"""Write examples/toy_results.csv: simulated results of 40 models on 24 items.

The data are drawn from a 2PL model with a fixed seed, with about 3 % of the cells removed, so
the file is synthetic and can be regenerated exactly:

    python examples/make_toy.py
"""

from pathlib import Path

from irtbench import simulate_responses, write_long

OUTPUT = Path(__file__).with_name("toy_results.csv")


def main() -> None:
    data, _ = simulate_responses(40, 24, kind="2pl", seed=2026, missing=0.03)
    write_long(data, OUTPUT)
    print(f"wrote {OUTPUT.name}: {data.n_models} models x {data.n_items} items, "
          f"{data.n_missing} missing cells")


if __name__ == "__main__":
    main()
