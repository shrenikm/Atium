import inspect
from typing import Any, Callable, Generic, TypeVar

import attr
from jax import hessian, jacfwd, jit

from atium.core.utils.attrs_utils import AttrsValidators
from atium.core.utils.custom_types import (
    OptimizationCostFn,
    OptimizationFn,
    OptimizationGradFn,
    OptimizationGradOrHessFn,
    OptimizationHessFn,
    Scalarf64,
    ScalarOrVectorNf64,
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
class DerivativeSplicedOptFn(Generic[TOptInput, TOptOutput]):
    """
    Container that holds a core optimization function, their gradient and hessian functions
    (Which can are auto-computed by JAX if not provided) and a params generation function
    that will be used to pass into the core and gradient functions.

    The core function can be a cost function or a constraint function (g <= 0, h = 0)
    The first argument (to the core and their derivative functions) is assumed to be a variable/vector of variables.
    The second argument can optionally be some params object that has all the data required to compute the
    function/derivative outputs.
    """

    core_fn: OptimizationFn[TOptInput, TOptOutput] = attr.ib(
        validator=AttrsValidators.num_args_validator(num_min_args=1, num_max_args=2)
    )
    use_jit: bool
    _grad_fn: OptimizationGradFn = attr.ib(validator=AttrsValidators.num_args_validator(num_min_args=1, num_max_args=2))
    _hess_fn: OptimizationHessFn = attr.ib(
        validator=AttrsValidators.num_args_validator(num_min_args=1, num_max_args=2),
    )
    _construct_params_fn: Callable[[ScalarOrVectorNf64], Any] | None = attr.ib(
        default=None,
        validator=attr.validators.optional(AttrsValidators.num_args_validator(num_min_args=1, num_max_args=1)),
    )
    _convexified_core_fn: OptimizationHessFn = attr.ib(
        init=False,
        validator=AttrsValidators.num_args_validator(num_min_args=2, num_max_args=3),
    )

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

    @_convexified_core_fn.default
    def _initialize_convexified_core_fn(self) -> OptimizationCostFn:
        """
        Returns a function that can be used to compute the convexified version of the core function.
        Utilizes the gradient and hessian.
        The convexified cost is the second order Taylor expansion around x
        f(new_x) = f(x) + f'(x)^T@(new_x - x) + (new_x - x)^T @ f''(x) @ (new_x - x)
        """

        def _ccf(
            x: ScalarOrVectorNf64,
            new_x: ScalarOrVectorNf64,
            params: Any | None = None,
        ) -> Scalarf64:
            if params is None:
                f0 = self.core_fn(x)
                omega = self._grad_fn(x)
                W = self._hess_fn(x)
            else:
                f0 = self.core_fn(x, params)
                omega = self._grad_fn(x, params)
                W = self._hess_fn(x, params)
            delta_x = new_x - x
            return f0 + omega @ delta_x + 0.5 * delta_x @ W @ delta_x

        return _ccf

    def construct_params(self, x: ScalarOrVectorNf64) -> Any:
        if self._construct_params_fn is None:
            return None
        else:
            return self._construct_params_fn(x)

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

    def convexified(self, x: ScalarOrVectorNf64, new_x: ScalarOrVectorNf64) -> Any:
        # Params construction is done internally, so we only need to pass in x.
        params = self.construct_params(x=x)
        if params is None:
            return self._convexified_core_fn(x, new_x)
        else:
            return self._convexified_core_fn(x, new_x, params)


DerivativeSplicedCostFn = DerivativeSplicedOptFn[ScalarOrVectorNf64, Scalarf64]
DerivativeSplicedConstraintsFn = DerivativeSplicedOptFn[ScalarOrVectorNf64, ScalarOrVectorNf64]
