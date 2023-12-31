import functools
import inspect
from re import T
from typing import Any, Callable, Generic, Optional, Protocol, TypeVar, Union

import attr
import jax.numpy as jnp
import numpy as np
from jax import grad, hessian, jacfwd, jacrev, jit, tree_map

from common.custom_types import (
    OptimizationFn,
    OptimizationGradFn,
    OptimizationGradOrHessFn,
    OptimizationHessFn,
    Scalarf64,
    ScalarOrVectorNf64,
)
from common.exceptions import AtiumOptError


def _probe_fn(fn: OptimizationFn) -> None:
    sig = inspect.signature(fn)
    num_fn_params = len(sig.parameters)

    if num_fn_params < 1 or num_fn_params > 2:
        raise AtiumOptError(
            """
            Function must have atleast one parameter and not more than two.
            Must be of the form f(x) or f(x, params)
            """
        )


def _get_jit_applied_fn(
    fn: OptimizationGradOrHessFn,
) -> OptimizationGradOrHessFn:
    # TODO: Doing this inspect twice. Unnecessary but maybe cleaner.
    sig = inspect.signature(fn)
    num_fn_params = len(sig.parameters)

    # Assuming the first argument is the scalar/vector of variables,
    # We can make all the other arguments static.
    static_argnums = tuple(range(1, num_fn_params))
    return jit(fn, static_argnums=static_argnums)


TOptInput = TypeVar("TOptInput", bound=ScalarOrVectorNf64)
TOptOutput = TypeVar("TOptOutput", bound=ScalarOrVectorNf64)


@attr.frozen
class DerivativeInitializedOptFn(Generic[TOptInput, TOptOutput]):
    """
    Container that holds a core optimization function, their gradient and hessian functions
    (Which can are auto-computed by JAX if not provided) and a params generation function
    that will be used to pass into the core and gradient functions.

    The core function can be a cost function or a constraint function (g <= 0, h = 0)
    The first argument (to the core and their derivative functions) is assumed to be a variable/vector of variables.
    The second argument can optionally be some params object that has all the data required to compute the
    function/derivative outputs.
    """

    core_fn: OptimizationFn[TOptInput, TOptOutput] = attr.ib()
    _grad_fn: OptimizationGradFn = attr.ib()
    _hess_fn: OptimizationHessFn = attr.ib()
    use_jit: bool = False

    @_grad_fn.default
    def _initialize_grad_fn(self) -> OptimizationGradFn:
        """
        The gradient in this case is a vector for scalar output functions and a matrix (jacobian)
        for vector output functions.
        Assumes that all the variables are provided in the first argument vector input.
        Note that we use jacfwd() because grad() doesn't work for vector output functions.
        """
        grad_fn = jacfwd(self.core_fn, argnums=0)
        if self.use_jit:
            grad_fn = _get_jit_applied_fn(fn=grad_fn)
        return grad_fn

    @_hess_fn.default
    def _initialize_hess_fn(self) -> OptimizationHessFn:
        """
        The hessian in this case is a matrix for scalar output functions and a tensor for vector
        output functions.
        Assumes that all the variables are provided in the first argument vector input.
        """
        hess_fn = hessian(self.core_fn, argnums=0)
        if self.use_jit:
            hess_fn = _get_jit_applied_fn(fn=hess_fn)
        return hess_fn

    @core_fn.validator
    def _validate_core_fn(self, _, value) -> None:
        """
        Making sure that the arguments to the core function are valid.
        """
        _probe_fn(fn=value)

    @_grad_fn.validator
    def _validate_grad_fn(self, _, value) -> None:
        """
        Making sure that the arguments to the grad function are valid.
        """
        _probe_fn(fn=value)

    @_hess_fn.validator
    def _validate_hess_fn(self, _, value) -> None:
        """
        Making sure that the arguments to the hess function are valid.
        """
        _probe_fn(fn=value)

    def construct_params(self, x: ScalarOrVectorNf64) -> Any:
        raise NotImplementedError

    def __call__(self, x: ScalarOrVectorNf64) -> Any:
        # Params construction is done internally, so we only need to pass in x.
        params = self.construct_params(x=x)
        if params is None:
            return self.core_fn(x)
        else:
            return self.core_fn(x, params)

    def grad(self, x: ScalarOrVectorNf64) -> Any:
        params = self.construct_params(x=x)
        if params is None:
            return self._grad_fn(x)
        else:
            return self._grad_fn(x, params)

    def hess(self, x: ScalarOrVectorNf64) -> Any:
        params = self.construct_params(x=x)
        if params is None:
            return self._hess_fn(x)
        else:
            return self._hess_fn(x, params)


DerivativeInitializedCostFn = DerivativeInitializedOptFn[ScalarOrVectorNf64, Scalarf64]
DerivativeInitializedConstraintsFn = DerivativeInitializedOptFn[
    ScalarOrVectorNf64, ScalarOrVectorNf64
]


if __name__ == "__main__":

    T = TypeVar("T")

    class A(Generic[T]):
        def f(self, t: T) -> T:
            print("A: ", t)
            return t

    AA = A[str]
    a = A[int]()
    a.f(3)

    aa = AA()
    aa.f("hi")
