import pytest
import attr
import numpy as np
from common.custom_types import MatrixNNf64, Scalarf64, VectorNf64
from common.exceptions import AtiumOptError
import jax.numpy as jnp

from common.optimization.tag_opt_fn import (
    TaggedAtiumOptFn,
    is_tagged_opt_fn,
    tag_atium_opt_fn,
)


@attr.frozen
class _Params:
    a: int
    b: float


def test_opt_fn_tagging():
    def f1(x: float) -> float:
        return x

    assert not is_tagged_opt_fn(fn=f1)

    @tag_atium_opt_fn
    def f2(x: float) -> float:
        return x

    assert is_tagged_opt_fn(fn=f2)

    @tag_atium_opt_fn(use_jit=False)
    def f3(x: float) -> float:
        return x

    assert is_tagged_opt_fn(fn=f3)


def test_no_arg_fn_tagging():
    with pytest.raises(AtiumOptError):

        @tag_atium_opt_fn
        def f() -> float:
            return 3.0


def test_custom_grad_hess_tagging():
    def _grad_fn(x: VectorNf64, params: _Params) -> VectorNf64:
        return params.a + x

    def _hess_fn(x: VectorNf64, params: _Params) -> MatrixNNf64:
        y = x.reshape(len(x), 1)
        return params.b * np.dot(y, y.T)

    with pytest.raises(AtiumOptError):
        # Exception check.
        @tag_atium_opt_fn(
            grad_fn=_grad_fn,
            hess_fn=_hess_fn,
            use_jit=True,
        )
        def f(x: VectorNf64, params: _Params) -> float:
            return params.b * np.dot(x, x)

    @tag_atium_opt_fn(
        grad_fn=_grad_fn,
        hess_fn=_hess_fn,
    )
    def f(x: VectorNf64, params: _Params) -> float:
        return params.b * np.dot(x, x)

    x = np.ones(3)
    params = _Params(1, 7.0)

    grad_vector = f.grad(x, params=params)
    hess_matrix = f.hess(x, params=params)
    np.testing.assert_equal(grad_vector, np.array([2.0, 2.0, 2.0]))
    np.testing.assert_equal(hess_matrix, np.full((3, 3), 7.0))


def test_tagged_jit_static_argnums():
    # Behind the scenes jits tatic_argnums should set the correct function arguments so that
    # JIT is too restrictive and we have more freedom to use loops, etc.
    @tag_atium_opt_fn(use_jit=True)
    def f(x: float, params: _Params) -> Scalarf64:
        y = jnp.array(0.0)
        for _ in range(params.a):
            y += x**2.0
        return y

    params = _Params(a=2, b=0.0)
    grad_value = f.grad(1.0, params=params).item()
    np.testing.assert_equal(grad_value, 4.0)


@pytest.mark.parametrize("use_jit", [True, False])
def test_scalar_input_scalar_output_tag(use_jit: bool):
    @tag_atium_opt_fn(use_jit=use_jit)
    def f(x: float) -> float:
        return x**3

    grad_value = f.grad(2.0).item()
    hess_value = f.hess(-2.0).item()
    np.testing.assert_equal(grad_value, 12.0)
    np.testing.assert_equal(hess_value, -12.0)


@pytest.mark.parametrize("use_jit", [True, False])
def test_scalar_input_vector_output_tag(use_jit: bool):
    @tag_atium_opt_fn(use_jit=use_jit)
    def f(x: float, param: _Params) -> Scalarf64:
        return x * jnp.ones(param.a)

    param = _Params(a=3, b=-3.0)
    grad_value = f.grad(2.0, param=param)
    hess_value = f.hess(2.0, param=param)
    np.testing.assert_equal(grad_value, np.array([1.0, 1.0, 1.0]))
    np.testing.assert_equal(hess_value, np.array([0.0, 0.0, 0.0]))


@pytest.mark.parametrize("use_jit", [True, False])
def test_vector_input_scalar_output_tag(use_jit: bool):
    @tag_atium_opt_fn(use_jit=use_jit)
    def f(x: VectorNf64, param: _Params) -> Scalarf64:
        n = len(x)
        Q = param.a * np.eye(n)
        return param.b + jnp.dot(x, jnp.dot(Q, x))

    param = _Params(a=7, b=1.0)
    x = np.array([1.0, 2.0, 3.0])
    grad_value = f.grad(x, param=param)
    hess_value = f.hess(x, param=param)
    np.testing.assert_equal(grad_value, np.array([14.0, 28.0, 42.0]))
    np.testing.assert_equal(hess_value, 14.0 * np.eye(3))


@pytest.mark.parametrize("use_jit", [True, False])
def test_vector_input_vector_output_tag(use_jit: bool):
    @tag_atium_opt_fn(use_jit=use_jit)
    def f(x: VectorNf64, param: _Params) -> VectorNf64:
        Q = param.a * np.eye(x.shape[0])
        return Q @ x + param.b

    param = _Params(a=2, b=-2.0)
    x = np.array([1.0, 2.0, 3.0])
    grad_value = f.grad(x, param=param)
    hess_value = f.hess(x, param=param)
    np.testing.assert_equal(grad_value, 2.0 * np.eye(3))
    np.testing.assert_equal(hess_value, np.zeros((3, 3, 3)))


if __name__ == "__main__":

    pytest.main(["-s", "-v", __file__])
