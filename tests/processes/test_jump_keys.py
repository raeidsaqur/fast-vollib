"""Jump laws consume separate keys, including under JAX's reuse checker."""

import numpy as np
import pytest

from fast_vollib.processes import Bates, ConstantVariance, LognormalJumps


def test_jump_counts_and_sizes_have_independent_jax_keys():
    jax = pytest.importorskip("jax")
    process = Bates(
        variance=ConstantVariance(),
        jumps=LognormalJumps(jump_intensity=2.0, mean_log_jump=-0.1, jump_volatility=0.2),
        drift=0.03,
    )
    with jax.debug_key_reuse(True):
        paths = process.sample(
            initial_state={"spot": 100.0, "variance": 0.04},
            time_grid=jax.numpy.array([0.0, 0.5, 1.0]),
            n_paths=8,
            rng=jax.random.key(19),
        )
    assert np.isfinite(np.asarray(paths)).all()
