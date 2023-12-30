import functools
from typing import Any, Callable, Optional, Protocol

import jax.numpy as jnp
import numpy as np
from jax import grad, hessian, jacfwd, jacrev, jit

from common.custom_types import (
    OptimizationFn,
    OptimizationGradFn,
    OptimizationHessFn,
    ScalarOrVectorN,
    VectorInputScalarOutputFn,
    VectorN,
)

GRAD_ATTR_NAME = "grad"
HESS_ATTR_NAME = "hess"
TAG_ATIUM_OPT_FN = "tag_opt_fn"


class TaggedAtiumOptFnKernel(Protocol):
    opt_fn_k: bool
    grad_fn: OptimizationGradFn
    hess_fn: OptimizationHessFn

    def __call__(
        self,
        x: ScalarOrVectorN,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> ScalarOrVectorN:
        raise NotImplementedError


def is_tagged_opt_fn(
    fn: Callable[[Any], Any],
) -> bool:
    if hasattr(fn, TAG_ATIUM_OPT_FN) and getattr(fn, TAG_ATIUM_OPT_FN):
        return True
    return False


def _tag_fn(
    fn: OptimizationFn,
    tag: str,
) -> None:
    setattr(fn, tag, True)


def _splice_grad(
    fn: OptimizationFn,
    grad_fn: Optional[OptimizationGradFn] = None,
    use_jit: bool = False,
) -> None:
    """
    Splices the given gradient function into the given scalar/vector output function.
    The gradient in this case is a vector for scalar output functions and a matrix (jacobian)
    for vector output functions.
    If no grad function is given, it uses JAX's autodiff grad to generate one.
    Assumes that all the variables are provided in the first argument vector input.
    Note that we use jacfwd() because grad() doesn't work for vector output functions.
    """
    if grad_fn is None:
        grad_fn = jacfwd(fn, argnums=0)
        if use_jit:
            grad_fn = jit(grad_fn)
    setattr(fn, GRAD_ATTR_NAME, grad_fn)


def _splice_hess(
    fn: OptimizationFn,
    hess_fn: Optional[OptimizationHessFn] = None,
    use_jit: bool = False,
) -> None:
    """
    Splices the given hessian function into the given scalar output function.
    The hessian in this case is a matrix for scalar output functions and a tensor for vector
    output functions.
    If a hess_fn isn't provided, uses JAX's autodiff hessian.
    Assumes that all the variables are provided in the first argument vector input.
    """
    if hess_fn is None:
        hess_fn = hessian(fn, argnums=0)
        if use_jit:
            hess_fn = jit(hess_fn)
    setattr(fn, HESS_ATTR_NAME, hess_fn)


def tag_atium_opt_fn(
    fn: Optional[VectorInputScalarOutputFn] = None,
    *,
    grad_fn: Optional[OptimizationGradFn] = None,
    hess_fn: Optional[OptimizationHessFn] = None,
    use_jit: bool = False,
):
    """
    Decorator to tag the function as an Atium optimization function.
    Splices it with gradient and hessian computation capabilities.
    Can be used for scalar and vector valued functions.
    Restrictions:
        1. All the variables are assumed to be given through a single vector as part of the first argument.
        2. use_jit = True would impose restrictions on the contents of the function being decorated. See the JAX documentation for more details.
    """

    def _tagged_fn(_fn):
        @functools.wraps(_fn)
        def _tagged_fn_wrapper(*args, **kwargs):
            return _fn(*args, **kwargs)

        # Tagging the function.
        _tag_fn(fn=_tagged_fn_wrapper, tag=TAG_ATIUM_OPT_FN)

        # Splicing the gradient function.
        _splice_grad(
            fn=_tagged_fn_wrapper,
            grad_fn=grad_fn,
            use_jit=use_jit,
        )

        # Splicing the hessian function.
        _splice_hess(
            fn=_tagged_fn_wrapper,
            hess_fn=hess_fn,
            use_jit=use_jit,
        )

        return _tagged_fn_wrapper

    if fn is None:
        return _tagged_fn
    else:
        # Making sure that the arguments make sense.
        return _tagged_fn(fn)


if __name__ == "__main__":

    @tag_atium_opt_fn(use_jit=True)
    # def f(x: jpt.ArrayLike) -> float:
    def f(x: VectorN) -> float:
        A = 6 * np.eye(len(x))
        return jnp.dot(x, jnp.dot(A, x))

    x = np.array([1.0, 2.0, 3.0])
    # print(f.__name__, f(x), f.a)
    print(f.__name__, f(x), f.tag_opt_fn)
    print(f.grad(x))
    print(f.hess(x))

    print(is_tagged_opt_fn(f))
