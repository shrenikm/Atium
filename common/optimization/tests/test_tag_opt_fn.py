import pytest
import attr
import numpy as np
from common.custom_types import MatrixNNf64, VectorNf64
from common.exceptions import AtiumOptError

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


def test_custom_grad_hess():
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
        def f1(x: VectorNf64, params: _Params) -> float:
            return params.b * np.dot(x, x)

    @tag_atium_opt_fn(
        grad_fn=_grad_fn,
        hess_fn=_hess_fn,
    )
    def f1(x: VectorNf64, params: _Params) -> float:
        return params.b * np.dot(x, x)

    x = np.ones(3)
    params = _Params(1, 7.0)

    grad_vector = f1.grad(x, params=params)
    hess_matrix = f1.hess(x, params=params)
    np.testing.assert_equal(grad_vector, [2.0, 2.0, 2.0])
    np.testing.assert_equal(hess_matrix, np.full((3, 3), 7.))


def test_opt_fn_tag_scalar_input_scalar_output():
    ...


if __name__ == "__main__":

    pytest.main(["-s", "-v", __file__])
