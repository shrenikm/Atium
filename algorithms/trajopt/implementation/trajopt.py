"""
Implementation of the TrajOpt algorithm.

See: https://rll.berkeley.edu/~sachin/papers/Schulman-IJRR2014.pdf
"""

from typing import Any, Protocol
import attr
from itertools import count

from common.custom_types import OptimizationConstraintFn, OptimizationCostFn, VectorNf64


@attr.frozen
class TrajOptOptFnParamsConstructor(Protocol):
    """
    Template for a class that can be used to construct params to pass to the optimization
    functions. This can incorporate problem specific constants, etc.

    This will then be implicitly passed to the cost, constraint and gradient functions internally.
    We assume each of the cost and constraint functions (along with their gradient and hessian functions)
    take in inputs of the form (x, params)

    params = constructor.construct_params_for_cost_fn(x=current_x)
    cost = cost_fn(x=current_x, params)
    """

    def construct_params_for_cost_fn(self, x: VectorNf64) -> Any:
        raise NotImplementedError

    def construct_params_for_inequality_constraints_fn(self, x: VectorNf64) -> Any:
        raise NotImplementedError

    def construct_params_for_equality_constraints_fn(self, x: VectorNf64) -> Any:
        raise NotImplementedError


@attr.frozen
class TrajOptParams:
    # Initial penalty coefficient
    mu_0: float
    # Initial trust region size
    s_0: float
    # Step acceptance parameter
    c: float
    # Trust region expansion and shrinkage factors
    tau_plus: float
    tau_minus: float
    # Penalty scaling factor
    k: float
    # Convergence thresholds
    f_tol: float
    x_tol: float
    # Constraint satisfaction threshold
    c_tol: float


@attr.frozen
class TrajOpt:

    params: TrajOptParams
    cost_fn: OptimizationCostFn
    inequality_constraints_fn: OptimizationConstraintFn
    equality_constraints_fn: OptimizationConstraintFn

    def solve(
        self,
        initial_guess_x: VectorNf64,
    ) -> None:
        for penalty_iter in count():
            for convexify_iter in count():
                ...
            ...
