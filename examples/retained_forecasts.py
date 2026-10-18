"""One simulation draw, repeated path queries and a separate marginal query."""

from datetime import datetime, timezone

import numpy as np

from fast_vollib.forecasting import ForecastQuery, ProcessDistribution
from fast_vollib.instruments import InstrumentRef
from fast_vollib.processes import GBM

origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
query = ForecastQuery(
    origin=origin, times=(0.0, 0.25, 1.0), convention="act_365", targets=("asset",), kind="path"
)
distribution = ProcessDistribution(
    process=GBM(drift=0.05, volatility=0.2),
    initial_state=100.0,
    underlier=InstrumentRef(identifier="ACME", currency="USD"),
    query=query,
    measure="physical",
    backend_metadata={"backend": "numpy"},
    conditioning_cutoff=origin,
    observations_digest="example:observations",
)
draws = distribution.draw(n_samples=16, rng=17)
first = draws.query(query)
second = draws.query(query)
np.testing.assert_array_equal(first.blocks["asset"].values, second.blocks["asset"].values)
np.testing.assert_array_equal(first.sample_ids, second.sample_ids)
view = first.to_scenario_view(target_id="asset")
assert view.scenario.states.shape == (16, 3, 1)
marginal_query = ForecastQuery(
    origin=origin, times=(1.0,), convention="act_365", targets=("asset",), kind="marginal"
)
marginal = draws.query(marginal_query)
np.testing.assert_array_equal(marginal.blocks["asset"].values, view.scenario.states[:, -1, :])
