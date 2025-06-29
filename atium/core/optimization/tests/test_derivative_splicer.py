from typing import Any

import attr
import jax.numpy as jnp
import numpy as np
import pytest

from atium.core.optimization.derivative_splicer import DerivativeSplicedOptFn
from atium.core.utils.custom_types import MatrixNNf64, OptimizationFn, Scalarf64, Vector3f64, VectorNf64


@attr.frozen
class _Params:
    a: int
    b: float


@pytest.fixture
def trivial_core_fn() -> OptimizationFn:
    def f(x: float) -> float:
        return x

    return f


@pytest.mark.parametrize("use_jit", [True, False])
def test_core_fn_validity(trivial_core_fn, use_jit: bool) -> None:
    def f1() -> float:
        return 1.0

    def f2(a, b, c) -> float:
        del a, b, c
        return 2.0

    with pytest.raises(ValueError):
        DerivativeSplicedOptFn(
            core_fn=f1,
            use_jit=use_jit,
        )

    with pytest.raises(ValueError):
        DerivativeSplicedOptFn(
            core_fn=f2,
            use_jit=use_jit,
        )

    # Valid.
    DerivativeSplicedOptFn(
        core_fn=trivial_core_fn,
        use_jit=use_jit,
    )


@pytest.mark.parametrize("use_jit", [True, False])
def test_invalid_custom_derivative_functions(trivial_core_fn, use_jit: bool) -> None:
    def _grad_fn() -> VectorNf64:
        return params.a + x

    def _hess_fn(x: VectorNf64, params: _Params, aux: Any) -> MatrixNNf64:
        y = x.reshape(len(x), 1)
        return params.b * np.dot(y, y.T)

    with pytest.raises(ValueError):
        DerivativeSplicedOptFn(
            core_fn=trivial_core_fn,
            use_jit=use_jit,
            grad_fn=_grad_fn,
        )

    with pytest.raises(ValueError):
        DerivativeSplicedOptFn(
            core_fn=trivial_core_fn,
            use_jit=use_jit,
            hess_fn=_hess_fn,
        )

    with pytest.raises(ValueError):
        DerivativeSplicedOptFn(
            core_fn=trivial_core_fn,
            use_jit=use_jit,
            grad_fn=_grad_fn,
            hess_fn=_hess_fn,
        )


@pytest.mark.parametrize("use_jit", [True, False])
def test_no_construct_params(trivial_core_fn, use_jit: bool) -> None:
    f_ds = DerivativeSplicedOptFn(
        core_fn=trivial_core_fn,
        use_jit=use_jit,
    )
    assert f_ds.construct_params(x=3.0) is None

    # We should still be able to compute the gradients and hessians
    # All these functions work without params now.
    x = 3.0
    f_value = f_ds(x=x)
    grad_value = f_ds.grad(x=x).item()
    hess_value = f_ds.hess(x=x).item()
    np.testing.assert_equal(f_value, 3.0)
    np.testing.assert_equal(grad_value, 1.0)
    np.testing.assert_equal(hess_value, 0.0)


@pytest.mark.parametrize("use_jit", [True, False])
def test_custom_derivative_functions(use_jit: bool) -> None:
    def _grad_fn(x: VectorNf64, params: _Params) -> VectorNf64:
        return params.a + x

    def _hess_fn(x: VectorNf64, params: _Params) -> MatrixNNf64:
        y = x.reshape(len(x), 1)
        return params.b * np.dot(y, y.T)

    def f(x: VectorNf64, params: _Params) -> float:
        return params.b * np.dot(x, x)

    def _construct_params(x: VectorNf64) -> _Params:
        return _Params(1, 7.0)

    f_ds = DerivativeSplicedOptFn(
        core_fn=f,
        use_jit=use_jit,
        grad_fn=_grad_fn,
        hess_fn=_hess_fn,
        construct_params_fn=_construct_params,
    )

    x = np.ones(3)

    grad_vector = f_ds.grad(x)
    hess_matrix = f_ds.hess(x)
    np.testing.assert_equal(grad_vector, np.array([2.0, 2.0, 2.0]))
    np.testing.assert_equal(hess_matrix, np.full((3, 3), 7.0))


def test_use_jit_static_argnums():
    # Behind the scenes jit's static_argnums should set the correct function arguments so that
    # JIT isn't too restrictive and we have more freedom to use loops, etc.
    def f(x: float, params: _Params) -> Scalarf64:
        y = jnp.array(0.0)
        for _ in range(params.a):
            y += x**2.0
        return y

    f_ds = DerivativeSplicedOptFn(
        core_fn=f,
        use_jit=True,
        construct_params_fn=lambda _: _Params(a=2, b=0.0),
    )

    grad_value = f_ds.grad(1.0).item()
    np.testing.assert_equal(grad_value, 4.0)


@pytest.mark.parametrize("use_jit", [True, False])
def test_scalar_input_scalar_output_tag(use_jit: bool):
    def f(x: float) -> float:
        return x**3

    f_ds = DerivativeSplicedOptFn(
        core_fn=f,
        use_jit=use_jit,
    )
    grad_value = f_ds.grad(2.0).item()
    hess_value = f_ds.hess(-2.0).item()
    np.testing.assert_equal(grad_value, 12.0)
    np.testing.assert_equal(hess_value, -12.0)


@pytest.mark.parametrize("use_jit", [True, False])
def test_scalar_input_vector_output_tag(use_jit: bool):
    def f(x: float, params: _Params) -> Scalarf64:
        return x * jnp.ones(params.a)

    f_ds = DerivativeSplicedOptFn(
        core_fn=f,
        use_jit=use_jit,
        construct_params_fn=lambda _: _Params(a=3, b=-3.0),
    )

    grad_value = f_ds.grad(2.0)
    hess_value = f_ds.hess(2.0)
    np.testing.assert_equal(grad_value, np.array([1.0, 1.0, 1.0]))
    np.testing.assert_equal(hess_value, np.array([0.0, 0.0, 0.0]))


@pytest.mark.parametrize("use_jit", [True, False])
def test_vector_input_scalar_output_tag(use_jit: bool):
    def f(x: VectorNf64, params: _Params) -> Scalarf64:
        n = len(x)
        Q = params.a * np.eye(n)
        return params.b + jnp.dot(x, jnp.dot(Q, x))

    f_ds = DerivativeSplicedOptFn(
        core_fn=f,
        use_jit=use_jit,
        construct_params_fn=lambda _: _Params(a=7, b=1.0),
    )

    x = np.array([1.0, 2.0, 3.0])
    grad_value = f_ds.grad(x)
    hess_value = f_ds.hess(x)
    np.testing.assert_equal(grad_value, np.array([14.0, 28.0, 42.0]))
    np.testing.assert_equal(hess_value, 14.0 * np.eye(3))


@pytest.mark.parametrize("use_jit", [True, False])
def test_vector_input_vector_output_tag(use_jit: bool):
    def f(x: VectorNf64, params: _Params) -> VectorNf64:
        Q = params.a * np.eye(x.shape[0])
        return Q @ x + params.b

    f_ds = DerivativeSplicedOptFn(
        core_fn=f,
        use_jit=use_jit,
        construct_params_fn=lambda _: _Params(a=2, b=-2.0),
    )

    x = np.array([1.0, 2.0, 3.0])
    grad_value = f_ds.grad(x)
    hess_value = f_ds.hess(x)
    np.testing.assert_equal(grad_value, 2.0 * np.eye(3))
    np.testing.assert_equal(hess_value, np.zeros((3, 3, 3)))


@pytest.mark.parametrize("use_jit", [True, False])
def test_realistic_constraint_fn_tag(use_jit: bool):
    """
    Test the gradient and hessian computations on a more realistic constraint function.
    """

    def f(z: VectorNf64, params: _Params) -> Vector3f64:
        """
        g = [xcos(theta) + ysin(theta); ax^3 + v^2 + bw; theta]
        """
        x, y, theta, v, w = z

        g = jnp.array(
            [
                x * jnp.cos(theta) + y * jnp.sin(theta),
                params.a * x**3 + v**2 + params.b * w,
                theta,
            ]
        )

        return g

    params = _Params(a=2, b=0.5)
    f_ds = DerivativeSplicedOptFn(
        core_fn=f,
        use_jit=use_jit,
        construct_params_fn=lambda _: params,
    )

    z = np.array([1.0, 2.0, np.pi / 3, 0.5, -1000.0])
    n = len(z)
    x, y, theta, v, w = z

    grad_value = f_ds.grad(z)
    hess_value = f_ds.hess(z)
    expected_grad_value = np.array(
        [
            [
                np.cos(theta),
                np.sin(theta),
                -x * np.sin(theta) + y * np.cos(theta),
                0.0,
                0.0,
            ],
            [3 * params.a * x**2, 0.0, 0.0, 2 * v, params.b],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )
    # Each 2d matrix of the expected 3d tensor.
    expected_hess_value_tensor_list = [np.zeros((3, n)) for _ in range(n)]
    expected_hess_value_tensor_list[0][0, 2] = -np.sin(theta)
    expected_hess_value_tensor_list[0][1, 0] = 6.0 * params.a * x
    expected_hess_value_tensor_list[1][0, 2] = np.cos(theta)
    expected_hess_value_tensor_list[2][0, 0] = -np.sin(theta)
    expected_hess_value_tensor_list[2][0, 1] = np.cos(theta)
    expected_hess_value_tensor_list[2][0, 2] = -x * np.cos(theta) - y * np.sin(theta)
    expected_hess_value_tensor_list[3][1, 3] = 2

    np.testing.assert_array_almost_equal(grad_value, expected_grad_value, decimal=6)
    for i in range(n):
        # Hessian has shape (5, 5, 3)
        np.testing.assert_array_almost_equal(hess_value[:, :, i], expected_hess_value_tensor_list[i], decimal=6)


@pytest.mark.parametrize("use_jit", [True, False])
def test_convexified_fn_on_convex_fn(use_jit: bool):
    """
    The implicit convexified cost computation of a convex function
    must return the same value.
    """

    def f(x: VectorNf64) -> Scalarf64:
        return jnp.dot(x, x)

    f_ds = DerivativeSplicedOptFn(
        core_fn=f,
        use_jit=use_jit,
    )

    rng = np.random.RandomState(7)
    x = rng.randn(5).round(6)
    new_x = rng.randn(5).round(6)

    core_value = f_ds(new_x)
    convexified_core_value = f_ds.convexified(x, new_x)

    np.testing.assert_almost_equal(core_value, convexified_core_value, decimal=6)


@pytest.mark.parametrize("use_jit", [True, False])
def test_convexified_fn_on_convex_fn(use_jit: bool):
    """
    The implicit convexified cost computation of a convex function
    must return the same value.
    """

    def f(z: VectorNf64, params: _Params) -> Scalarf64:
        x, y, theta = z
        return jnp.array(
            params.a * x * jnp.cos(theta) + params.b * y * jnp.sin(theta),
            dtype=jnp.float32,
        )

    params = _Params(a=2, b=0.5)
    f_ds = DerivativeSplicedOptFn(
        core_fn=f,
        use_jit=use_jit,
        construct_params_fn=lambda _: params,
    )

    rng = np.random.RandomState(7)
    z = rng.randn(3).round(6)
    new_z = rng.randn(3).round(6)

    convexified_core_value = f_ds.convexified(z, new_z)

    # Expected convexified cost.
    x, y, theta = z
    omega = np.array(
        [
            params.a * np.cos(theta),
            params.b * np.sin(theta),
            -params.a * x * np.sin(theta) + params.b * y * np.cos(theta),
        ]
    )
    W = np.array(
        [
            [0.0, 0.0, -params.a * np.sin(theta)],
            [0.0, 0.0, params.b * np.cos(theta)],
            [
                -params.a * np.sin(theta),
                params.b * np.cos(theta),
                -params.a * x * np.cos(theta) - params.b * y * np.sin(theta),
            ],
        ]
    )
    expected_convexified_core_value = f_ds(z) + np.dot(omega, new_z - z) + 0.5 * np.dot(np.dot(new_z - z, W), new_z - z)
    np.testing.assert_almost_equal(convexified_core_value, expected_convexified_core_value, decimal=6)


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
