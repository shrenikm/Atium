import functools
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jacfwd, jacrev, jit

from common.custom_types import (
    VectorInputMatrixOutputHessFn,
    VectorInputScalarOutputFn,
    VectorInputVectorOutputGradFn,
)

GRAD_ATTR_NAME = "grad"
HESS_ATTR_NAME = "hess"
VISO_KERNEL_TAG = "VISO_K"


def atium_opt_viso_kernel(
    fn: Optional[VectorInputScalarOutputFn] = None,
    *,
    grad_fn: Optional[VectorInputVectorOutputGradFn] = None,
    hessian_fn: Optional[VectorInputMatrixOutputHessFn] = None,
    use_jit: bool = False,
):
    """
    Kernel for a single vector input, scalar output function to be used in optimization.
    Generally useful for cost functions of the form J = f(x)
    """

    def _core(_fn):
        @functools.wraps(_fn)
        def _wrapper(*args, **kwargs):
            return _fn(*args, **kwargs)

        nonlocal grad_fn
        if grad_fn is None:
            grad_fn = grad(_wrapper)
            if use_jit:
                grad_fn = jit(grad_fn)
        setattr(_wrapper, GRAD_ATTR_NAME, grad_fn)
        return _wrapper

    if fn is None:
        return _core
    else:
        # Making sure that the arguments make sense.
        return _core(fn)


if __name__ == "__main__":

    @atium_opt_viso_kernel(use_jit=True)
    # @atium_opt_viso_kernel
    def f(x) -> float:
        print("in f")
        A = 6 * np.eye(len(x))
        return jnp.dot(x, jnp.dot(A, x))

    x = np.array([1.0, 2.0, 3.0])
    # print(f.__name__, f(x), f.a)
    print(f.__name__, f(x), f.grad(x))
