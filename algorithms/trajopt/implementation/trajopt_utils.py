import attr
from typing import Optional, Protocol, Any

from common.custom_types import VectorNf64


@attr.frozen
class TrajOptOptFnParamsConstructor(Protocol):
    """
    Template for a class that can be used to construct params to pass to the optimization
    functions. This can incorporate problem specific constants, etc.

    This will then be implicitly passed to the cost, constraint and gradient functions internally.
    We assume each of the cost and constraint functions (along with their gradient and hessian functions)
    take in inputs of the form (x, params). If the constructor is not defined, then we assume that inputs
    only take in x.

    params = constructor.construct_params_for_cost_fn(x=current_x)
    cost = cost_fn(x=current_x, params)
    """

    def construct_params_for_cost_fn(self, x: VectorNf64) -> Optional[Any]:
        raise NotImplementedError

    def construct_params_for_linear_inequality_constraints_fn(
        self, x: VectorNf64
    ) -> Optional[Any]:
        raise NotImplementedError

    def construct_params_for_non_linear_inequality_constraints_fn(
        self, x: VectorNf64
    ) -> Optional[Any]:
        raise NotImplementedError

    def construct_params_for_linear_equality_constraints_fn(
        self, x: VectorNf64
    ) -> Optional[Any]:
        raise NotImplementedError

    def construct_params_for_non_linear_equality_constraints_fn(
        self, x: VectorNf64
    ) -> Optional[Any]:
        raise NotImplementedError


@attr.frozen
class DefaultTrajOptFnParamsConstructor(Protocol):
    """
    Default fn params constructor that returns all Nones
    """

    def construct_params_for_cost_fn(self, x: VectorNf64) -> Optional[Any]:
        return None

    def construct_params_for_linear_inequality_constraints_fn(
        self, x: VectorNf64
    ) -> Optional[Any]:
        return None

    def construct_params_for_non_linear_inequality_constraints_fn(
        self, x: VectorNf64
    ) -> Optional[Any]:
        return None

    def construct_params_for_linear_equality_constraints_fn(
        self, x: VectorNf64
    ) -> Optional[Any]:
        return None

    def construct_params_for_non_linear_equality_constraints_fn(
        self, x: VectorNf64
    ) -> Optional[Any]:
        return None
