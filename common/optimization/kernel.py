import functools
import inspect
from typing import Callable, Optional, get_args

import jax
import jax.numpy as jnp
import jax.typing as jpt
import numpy as np
import numpy.typing as npt
from jax import grad, jacfwd, jacrev, jit

from common.custom_types import (
    Vector,
    VectorInputMatrixOutputHessFn,
    VectorInputScalarOutputFn,
    VectorInputVectorOutputGradFn,
)

GRAD_ATTR_NAME = "grad"
HESS_ATTR_NAME = "hess"
VISO_KERNEL_TAG = "t_viso_k"


def atium_opt_viso_kernel(
    fn: Optional[VectorInputScalarOutputFn] = None,
    *,
    grad_fn: Optional[VectorInputVectorOutputGradFn] = None,
    hess_fn: Optional[VectorInputMatrixOutputHessFn] = None,
    use_jit: bool = False,
):
    """
    Kernel for a single vector input, scalar output function to be used in optimization.
    Generally useful for cost functions of the form J = f(x)
    """

    def _probe_fn(_fn: VectorInputScalarOutputFn) -> None:
        assert inspect.isfunction(_fn)

        sig = inspect.signature(_fn)
        print(sig.parameters)
        first_attr_key, first_attr_params = next(iter(sig.parameters.items()))
        ann = first_attr_params.annotation
        print(ann)
        allowed_ann = get_args(jpt.ArrayLike) + (npt.ArrayLike,)
        print(allowed_ann)
        assert ann == jpt.ArrayLike or ann in allowed_ann
        print(first_attr_key, first_attr_params)
        print(type(first_attr_key), type(first_attr_params))

    def _tag_fn(_fn: VectorInputScalarOutputFn) -> None:
        setattr(_fn, VISO_KERNEL_TAG, True)

    def _splice_grad(_fn: VectorInputScalarOutputFn) -> None:
        """
        If a grad_fn isn't provided, uses JAX's autodiff grad.
        Assumes that all the variables are provided in the first argument vector input.
        """
        nonlocal grad_fn
        if grad_fn is None:
            grad_fn = grad(_fn, argnums=0)
            if use_jit:
                grad_fn = jit(grad_fn)
        setattr(_fn, GRAD_ATTR_NAME, grad_fn)

    def _splice_hess(_fn: VectorInputScalarOutputFn) -> None:
        """
        If a grad_fn isn't provided, uses JAX's autodiff hessian.
        Assumes that all the variables are provided in the first argument vector input.
        """
        nonlocal hess_fn
        if hess_fn is None:
            hess_fn = jacfwd(grad(_fn, argnums=0), argnums=0)
            if use_jit:
                hess_fn = jit(hess_fn)
        setattr(_fn, HESS_ATTR_NAME, hess_fn)

    def _tagged_fn(_fn):
        @functools.wraps(_fn)
        def _tagged_fn_wrapper(*args, **kwargs):
            return _fn(*args, **kwargs)

        # First we check the function inputs.
        _probe_fn(_fn=_tagged_fn_wrapper)

        # Tagging the function.
        _tag_fn(_fn=_tagged_fn_wrapper)

        # Splicing the gradient function.
        _splice_grad(_fn=_tagged_fn_wrapper)

        # Splicing the hessian function.
        _splice_hess(_fn=_tagged_fn_wrapper)

        return _tagged_fn_wrapper

    if fn is None:
        return _tagged_fn
    else:
        # Making sure that the arguments make sense.
        return _tagged_fn(fn)


if __name__ == "__main__":

    @atium_opt_viso_kernel(use_jit=True)
    # @atium_opt_viso_kernel
    # def f(x: jpt.ArrayLike) -> float:
    def f(x: Vector) -> float:
        A = 6 * np.eye(len(x))
        return jnp.dot(x, jnp.dot(A, x))

    x = np.array([1.0, 2.0, 3.0])
    # print(f.__name__, f(x), f.a)
    print(f.__name__, f(x), f.t_viso_k)
    print(f.grad(x))
    print(f.hess(x))
