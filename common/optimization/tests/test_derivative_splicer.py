import attr
import jax.numpy as jnp
import numpy as np
import pytest

from common.custom_types import MatrixNNf64, Scalarf64, Vector3f64, VectorNf64
from common.exceptions import AtiumOptError
from common.optimization.derivative_splicer import DerivativeSplicedOptFn


@attr.frozen
class _Params:
    a: int
    b: float


def test_opt_fn_tagging():
    def f1(x: float) -> float:
        return x

    assert not is_tagged_opt_fn(fn=f1)

    @tag_opt_fn
    def f2(x: float) -> float:
        return x

    assert is_tagged_opt_fn(fn=f2)

    @tag_opt_fn(use_jit=False)
    def f3(x: float) -> float:
        return x

    assert is_tagged_opt_fn(fn=f3)


def test_core_fn_validity():
    def f1(x: float) -> float:
        return x

    f1_ds = DerivativeSplicedOptFn(
        core_fn=f1,
    )


# def test_no_arg_fn_tagging():
#    with pytest.raises(AtiumOptError):
#
#        @tag_opt_fn
#        def f() -> float:
#            return 3.0
#
#
# def test_custom_grad_hess_tagging():
#    def _grad_fn(x: VectorNf64, params: _Params) -> VectorNf64:
#        return params.a + x
#
#    def _hess_fn(x: VectorNf64, params: _Params) -> MatrixNNf64:
#        y = x.reshape(len(x), 1)
#        return params.b * np.dot(y, y.T)
#
#    with pytest.raises(AtiumOptError):
#        # Exception check.
#        @tag_opt_fn(
#            grad_fn=_grad_fn,
#            hess_fn=_hess_fn,
#            use_jit=True,
#        )
#        def f(x: VectorNf64, params: _Params) -> float:
#            return params.b * np.dot(x, x)
#
#    @tag_opt_fn(
#        grad_fn=_grad_fn,
#        hess_fn=_hess_fn,
#    )
#    def f(x: VectorNf64, params: _Params) -> float:
#        return params.b * np.dot(x, x)
#
#    x = np.ones(3)
#    params = _Params(1, 7.0)
#
#    grad_vector = f.grad(x, params=params)
#    hess_matrix = f.hess(x, params=params)
#    np.testing.assert_equal(grad_vector, np.array([2.0, 2.0, 2.0]))
#    np.testing.assert_equal(hess_matrix, np.full((3, 3), 7.0))
#
#
# def test_tagged_jit_static_argnums():
#    # Behind the scenes jits tatic_argnums should set the correct function arguments so that
#    # JIT is too restrictive and we have more freedom to use loops, etc.
#    @tag_opt_fn(use_jit=True)
#    def f(x: float, params: _Params) -> Scalarf64:
#        y = jnp.array(0.0)
#        for _ in range(params.a):
#            y += x**2.0
#        return y
#
#    params = _Params(a=2, b=0.0)
#    grad_value = f.grad(1.0, params=params).item()
#    np.testing.assert_equal(grad_value, 4.0)
#
#
# @pytest.mark.parametrize("use_jit", [True, False])
# def test_scalar_input_scalar_output_tag(use_jit: bool):
#    @tag_opt_fn(use_jit=use_jit)
#    def f(x: float) -> float:
#        return x**3
#
#    grad_value = f.grad(2.0).item()
#    hess_value = f.hess(-2.0).item()
#    np.testing.assert_equal(grad_value, 12.0)
#    np.testing.assert_equal(hess_value, -12.0)
#
#
# @pytest.mark.parametrize("use_jit", [True, False])
# def test_scalar_input_vector_output_tag(use_jit: bool):
#    @tag_opt_fn(use_jit=use_jit)
#    def f(x: float, params: _Params) -> Scalarf64:
#        return x * jnp.ones(params.a)
#
#    params = _Params(a=3, b=-3.0)
#    grad_value = f.grad(2.0, params=params)
#    hess_value = f.hess(2.0, params=params)
#    np.testing.assert_equal(grad_value, np.array([1.0, 1.0, 1.0]))
#    np.testing.assert_equal(hess_value, np.array([0.0, 0.0, 0.0]))
#
#
# @pytest.mark.parametrize("use_jit", [True, False])
# def test_vector_input_scalar_output_tag(use_jit: bool):
#    @tag_opt_fn(use_jit=use_jit)
#    def f(x: VectorNf64, params: _Params) -> Scalarf64:
#        n = len(x)
#        Q = params.a * np.eye(n)
#        return params.b + jnp.dot(x, jnp.dot(Q, x))
#
#    params = _Params(a=7, b=1.0)
#    x = np.array([1.0, 2.0, 3.0])
#    grad_value = f.grad(x, params=params)
#    hess_value = f.hess(x, params=params)
#    np.testing.assert_equal(grad_value, np.array([14.0, 28.0, 42.0]))
#    np.testing.assert_equal(hess_value, 14.0 * np.eye(3))
#
#
# @pytest.mark.parametrize("use_jit", [True, False])
# def test_vector_input_vector_output_tag(use_jit: bool):
#    @tag_opt_fn(use_jit=use_jit)
#    def f(x: VectorNf64, params: _Params) -> VectorNf64:
#        Q = params.a * np.eye(x.shape[0])
#        return Q @ x + params.b
#
#    params = _Params(a=2, b=-2.0)
#    x = np.array([1.0, 2.0, 3.0])
#    grad_value = f.grad(x, params=params)
#    hess_value = f.hess(x, params=params)
#    np.testing.assert_equal(grad_value, 2.0 * np.eye(3))
#    np.testing.assert_equal(hess_value, np.zeros((3, 3, 3)))
#
#
# @pytest.mark.parametrize("use_jit", [True, False])
# def test_realistic_constraint_fn_tag(use_jit: bool):
#    """
#    Test the gradient and hessian computations on a more realistic constraint function.
#    """
#
#    @tag_opt_fn(use_jit=use_jit)
#    def f(z: VectorNf64, params: _Params) -> Vector3f64:
#        """
#        g = [xcos(theta) + ysin(theta); ax^3 + v^2 + bw; theta]
#        """
#        x, y, theta, v, w = z
#
#        # g = jnp.zeros(3)
#        # g.at[0].set(x * jnp.cos(theta) + y * jnp.sin(theta))
#        # g.at[1].set(params.a * x**3 + v**2 + params.b * w)
#        # g.at[2].set(theta)
#
#        g = jnp.array(
#            [
#                x * jnp.cos(theta) + y * jnp.sin(theta),
#                params.a * x**3 + v**2 + params.b * w,
#                theta,
#            ]
#        )
#
#        return g
#
#    params = _Params(a=2, b=0.5)
#    z = np.array([1.0, 2.0, np.pi / 3, 0.5, -1000.0])
#    n = len(z)
#    x, y, theta, v, w = z
#
#    grad_value = f.grad(z, params=params)
#    hess_value = f.hess(z, params=params)
#    expected_grad_value = np.array(
#        [
#            [
#                np.cos(theta),
#                np.sin(theta),
#                -x * np.sin(theta) + y * np.cos(theta),
#                0.0,
#                0.0,
#            ],
#            [3 * params.a * x**2, 0.0, 0.0, 2 * v, params.b],
#            [0.0, 0.0, 1.0, 0.0, 0.0],
#        ]
#    )
#    # Each 2d matrix of the expected 3d tensor.
#    expected_hess_value_tensor_list = [np.zeros((3, n)) for _ in range(n)]
#    expected_hess_value_tensor_list[0][0, 2] = -np.sin(theta)
#    expected_hess_value_tensor_list[0][1, 0] = 6.0 * params.a * x
#    expected_hess_value_tensor_list[1][0, 2] = np.cos(theta)
#    expected_hess_value_tensor_list[2][0, 0] = -np.sin(theta)
#    expected_hess_value_tensor_list[2][0, 1] = np.cos(theta)
#    expected_hess_value_tensor_list[2][0, 2] = -x * np.cos(theta) - y * np.sin(theta)
#    expected_hess_value_tensor_list[3][1, 3] = 2
#
#    np.testing.assert_array_almost_equal(grad_value, expected_grad_value, decimal=6)
#    for i in range(n):
#        # TODO: The hessian returned has dimensions 3x5x5. Check if there is some way of changing it to 5x3x5
#        #  without reshaping it manually.
#        np.testing.assert_array_almost_equal(
#            hess_value[:, :, i], expected_hess_value_tensor_list[i], decimal=6
#        )


if __name__ == "__main__":

    pytest.main(["-s", "-v", __file__])
