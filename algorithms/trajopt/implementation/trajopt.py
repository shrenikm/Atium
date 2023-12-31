"""
Implementation of the TrajOpt algorithm.

See: https://rll.berkeley.edu/~sachin/papers/Schulman-IJRR2014.pdf
"""
import numpy as np

from typing import Any, Optional, Protocol
import attr
from itertools import count
from algorithms.trajopt.implementation.trajopt_utils import (
    DefaultTrajOptFnParamsConstructor,
    TrajOptOptFnParamsConstructor,
)

from common.custom_types import VectorNf64
from common.optimization.tag_opt_fn import TaggedOptConstraintsFn, TaggedOptCostFn


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


@attr.define
class TrajOpt:

    params: TrajOptParams

    cost_fn: TaggedOptCostFn
    non_linear_inequality_constraints_fn: Optional[TaggedOptConstraintsFn] = None
    linear_inequality_constraints_fn: Optional[TaggedOptConstraintsFn] = None
    non_linear_equality_constraints_fn: Optional[TaggedOptConstraintsFn] = None
    linear_equality_constraints_fn: Optional[TaggedOptConstraintsFn] = None

    problem_params_constructor: TrajOptOptFnParamsConstructor = attr.ib(
        factory=DefaultTrajOptFnParamsConstructor
    )

    def convexify_problem(
        self,
        x: VectorNf64,
    ) -> None:
        n = len(x)
        # Computing the gradient and hessian of the cost function.
        # f = cost function, g = inequality constraints, h = equality constraints
        f_params = self.problem_params_constructor.construct_params_for_cost_fn(x=x)
        lg_params = self.problem_params_constructor.construct_params_for_linear_inequality_constraints_fn(
            x=x
        )
        nlg_params = self.problem_params_constructor.construct_params_for_non_linear_inequality_constraints_fn(
            x=x
        )
        lh_params = self.problem_params_constructor.construct_params_for_linear_equality_constraints_fn(
            x=x
        )
        nlh_params = self.problem_params_constructor.construct_params_for_non_linear_equality_constraints_fn(
            x=x
        )
        f_args = (x, f_params) if f_params is not None else (x,)
        lg_args = (x, lg_params) if lg_params is not None else (x,)
        nlg_args = (x, nlg_params) if nlg_params is not None else (x,)
        lh_args = (x, lh_params) if lh_params is not None else (x,)
        nlh_args = (x, nlh_params) if nlh_params is not None else (x,)

        # Constraints (These are all linearized, hence we only require the gradients).
        # The gradients are still matrices (as they're vector output) so they are represented by w_f.
        # Also computing the values at the current x as for everything other than the cost function, we need the
        # first term in the Taylor expansion.
        # The taylor expansion for constraints will be:
        # g(x) = g(x0) + W_g(x - x0) <= 0
        # => W_gx <= W_gx0 - g(x0)
        # W_hx + h(x0) - W_hx0 = 0
        # These can then be converted into:
        # A_gx + b_g <= 0
        # A_hx = b_h
        # Where x0 just refers to the current x about which we are linearizing.

        # Setting up the A matrix for OSQP
        # l <= Ax <= u
        # We compute the A for each linear constraint first and stack them up.
        # The non linear constraints have a dependency on the slack variables introduced so these
        # will be added later after A has been expanded.
        A = np.empty((0, n), dtype=np.float64)
        lb, ub = np.empty(0), np.empty(0)
        if self.linear_inequality_constraints_fn is not None:
            W_lg = self.linear_inequality_constraints_fn.grad(lg_args)
            lg0 = self.linear_inequality_constraints_fn(lg_args)

            A_lg = W_lg
            ub_lg = W_lg @ x - lg0

            assert lg0.ndim == 1
            A = np.vstack((A, A_lg))
            ub = np.hstack((ub, ub_lg))
            # Lower limits are all -inf for this set.
            lb = np.hstack((lb, len(ub_lg)))

        if self.linear_equality_constraints_fn is not None:
            W_lh = self.linear_equality_constraints_fn.grad(lh_args)
            lh0 = self.linear_equality_constraints_fn(lh_args)

            A_lh = W_lh
            b_lh = W_lh @ x - lh0

            assert lh0.ndim == 1

            # As it is an equality constraint, upper and lower bounds are both equal
            A = np.vstack((A, A_lh))
            lb = np.hstack((lb, b_lh))
            ub = np.hstack((ub, b_lh))

        W_nlg = self.non_linear_inequality_constraints_fn.grad(nlg_args)
        nlg0 = self.non_linear_inequality_constraints_fn(nlg_args)

        W_nlh = self.non_linear_equality_constraints_fn.grad(nlh_args)
        nlh0 = self.non_linear_equality_constraints_fn(nlh_args)

        # Computing the necessary gradients and hessians for the current x.
        # Cost function. Gradient vector by omega and hessian matrix by W
        omega_f = self.cost_fn.grad(*f_args)
        W_f = self.cost_fn.hess(*f_args)

    def solve(
        self,
        initial_guess_x: VectorNf64,
    ) -> None:
        for penalty_iter in count():
            for convexify_iter in count():
                ...
            ...
