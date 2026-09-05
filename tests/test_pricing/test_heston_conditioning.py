"""High-precision checks for the small-vol-of-vol characteristic function."""

import mpmath as mp
import numpy as np
import pytest

from fast_vollib.pricing import heston_characteristic_function


@pytest.mark.parametrize("xi", [1e-3, 1e-5, 1e-8, 1e-12])
@pytest.mark.parametrize("rho", [-0.7, 0.0, 0.7])
@pytest.mark.parametrize("maturity", [0.01, 1.0, 30.0])
def test_small_volatility_against_high_precision(xi, rho, maturity):
    args = np.array([0, -1j, 0.1, 1.0, 5.0 - 0.5j, 30.0 - 1j])
    with mp.workdps(80):
        sigma, correlation, T = map(mp.mpf, [xi, rho, maturity])
        expected = []
        for value in args:
            u = mp.mpc(complex(value))
            beta = 2 - correlation * sigma * 1j * u
            d = mp.sqrt(beta**2 + sigma**2 * (u * u + 1j * u))
            g = (beta - d) / (beta + d)
            e = mp.exp(-d * T)
            D = (beta - d) / sigma**2 * (1 - e) / (1 - g * e)
            C = 2 * mp.mpf(0.06) / sigma**2 * ((beta - d) * T - 2 * mp.log((1 - g * e) / (1 - g)))
            expected.append(complex(mp.exp(C + D * mp.mpf(0.04))))
    actual = heston_characteristic_function(
        args,
        maturity=maturity,
        v0=0.04,
        kappa=2.0,
        theta=0.06,
        vol_of_vol=xi,
        rho=rho,
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-14)


def test_martingale_argument_with_nonpositive_beta():
    with np.errstate(all="raise"):
        actual = heston_characteristic_function(
            np.array([0, -1j]),
            maturity=2.0,
            v0=0.04,
            kappa=0.1,
            theta=0.04,
            vol_of_vol=1.0,
            rho=0.8,
        )
    np.testing.assert_array_equal(actual, [1, 1])


@pytest.mark.parametrize("xi", [0.3, 1e-8])
@pytest.mark.parametrize("u", [1.0, 1.0 - 0.5j, 0.0, -1j])
@pytest.mark.parametrize("wrap", [lambda x: x, np.complex128, np.asarray])
def test_scalar_transform_preserves_the_numpy_scalar_return(xi, u, wrap):
    parameters = dict(maturity=1.0, v0=0.04, kappa=2.0, theta=0.04, vol_of_vol=xi, rho=-0.7)
    scalar = heston_characteristic_function(wrap(u), **parameters)
    array = heston_characteristic_function([u], **parameters)
    assert isinstance(scalar, np.complex128)
    assert isinstance(array, np.ndarray)
    assert array.shape == (1,)
    np.testing.assert_array_equal(scalar, array[0])


@pytest.mark.parametrize("xi", [0.3, 1e-8])
@pytest.mark.parametrize("shape", [(0,), (1,), (2, 3)])
def test_array_transform_preserves_shape(xi, shape):
    actual = heston_characteristic_function(
        np.ones(shape),
        maturity=1.0,
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        vol_of_vol=xi,
        rho=-0.7,
    )
    assert isinstance(actual, np.ndarray)
    assert actual.shape == shape
    assert actual.dtype == np.complex128
