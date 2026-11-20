# IV stress-set harness (v0.2.5 validation)

Use these scripts to reproduce the implied-volatility development and stress-set results.
`scripts/validation_matrix.py` scores the canonical grid, varied-parameter sample and fixtures.
The scripts below also record which rows cause an error signal.

| Script | Purpose |
|---|---|
| `make_devset.py`, `make_confirmset.py` | build the development set and the fresh-seed stress sets (inputs; references come from `scripts/reference_inverse.py`) |
| `evalsets.py` | load a set (inputs + exact-quote reference) for scoring |
| `run_route.py` | run one IV route on one set: one call per contract group, then every NaN row alone under `on_error="raise"` to attribute the signal per row |
| `score.py`, `scoring.py` | score route outputs under the frozen protocol in `tests/fixtures/iv_adjudication/protocol.json` |
| `compare_routes.py` | compare routes on the same set |
| `coverage_floor.py` | finite coverage and per-row losses against a baseline release |
| `iv_timing.py` | back-to-back IV timing of two source trees on frozen inputs |

Use the `stress_sets/` directory from the evidence archive.
It contains the input and reference CSV files for `devset`, `confirm` and `confirm2`.
Set `FV_IV_SETS` to that directory. Run these commands from the repository root.

```bash
export FV_IV_SETS=/path/to/stress_sets
PYTHONPATH=src python scripts/iv_stress/run_route.py --set confirm2 --route jackel_numpy --out out/confirm2__jackel_numpy_cpu.csv
python scripts/iv_stress/score.py --set confirm2 --routes-dir out --out out/score_confirm2.json --fails out/fails_confirm2.csv
```

`coverage_floor.py` compares the results with the v0.2.4 results.
To generate the baseline results, set `PYTHONPATH` to the `src/` directory in the v0.2.4 checkout.
