"""
Implementation of the TrajOpt algorithm.

See: https://rll.berkeley.edu/~sachin/papers/Schulman-IJRR2014.pdf
"""
from itertools import count
from typing import Any, Optional, Protocol

import attr
import numpy as np

from algorithms.trajopt.implementation.trajopt_utils import (
    DefaultTrajOptFnParamsConstructor,
    TrajOptOptFnParamsConstructor,
)
from common.custom_types import VectorNf64
from common.optimization.constructs import QPInputs
from common.optimization.derivative_splicer import (
    DerivativeSplicedConstraintsFn,
    DerivativeSplicedCostFn,
)


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

    cost_fn: DerivativeSplicedCostFn
    linear_inequality_constraints_fn: Optional[DerivativeSplicedConstraintsFn] = None
    linear_equality_constraints_fn: Optional[DerivativeSplicedConstraintsFn] = None
    non_linear_inequality_constraints_fn: Optional[
        DerivativeSplicedConstraintsFn
    ] = None
    non_linear_equality_constraints_fn: Optional[DerivativeSplicedConstraintsFn] = None

    def convexify_problem(
        self,
        x: VectorNf64,
        s: float,
        mu: float,
    ) -> QPInputs:
        assert s > 0.0
        n = len(x)
        # Computing the gradient and hessian of the cost function.
        # f = cost function, g = inequality constraints, h = equality constraints

        # Constraints (These are all linearized, hence we only require the gradients).
        # The gradients are still matrices (as they're vector output) so they are represented by w_f.
        # Also computing the values at the current x as for everything other than the cost function, we need the
        # first term in the Taylor expansion.
        # The taylor expansion for constraints will be:
        # g(x) = g(x0) + W_g@(x - x0) <= 0
        # => W_gx <= W_g@x0 - g(x0)
        # W_h@x + h(x0) - W_h@x0 = 0
        # These can then be converted into the forms:
        # A_g@x <= b_g
        # A_h@x = b_h
        # Where x0 just refers to the current x about which we are linearizing.

        # Setting up the A matrix for OSQP
        # l <= Ax <= u
        # We compute the A for each linear constraint first and stack them up.
        # The non linear constraints have a dependency on the slack variables introduced so these
        # will be added later after A has been expanded.
        A = np.empty((0, n), dtype=np.float64)
        lb, ub = np.empty(0), np.empty(0)
        if self.linear_inequality_constraints_fn is not None:
            lg0 = self.linear_inequality_constraints_fn(x)
            W_lg = self.linear_inequality_constraints_fn.grad(x)

            A_lg = W_lg
            ub_lg = W_lg @ x - lg0
            # Lower limits are all -inf for this set.
            lb_lg = np.full(len(ub_lg), fill_value=-np.inf)

            assert lg0.ndim == 1
            A = np.vstack((A, A_lg))
            lb = np.hstack((lb, lb_lg))
            ub = np.hstack((ub, ub_lg))

        if self.linear_equality_constraints_fn is not None:
            lh0 = self.linear_equality_constraints_fn(x)
            W_lh = self.linear_equality_constraints_fn.grad(x)

            A_lh = W_lh
            b_lh = W_lh @ x - lh0

            assert lh0.ndim == 1

            # As it is an equality constraint, upper and lower bounds are both equal
            A = np.vstack((A, A_lh))
            lb = np.hstack((lb, b_lh))
            ub = np.hstack((ub, b_lh))

        # Before we do the same for the non linear constraints, we first need to compute the
        # number of non-linear constraints. This will tell us how many slack variables we need
        # to add to the problem.
        # Each inequality constraint is converted to a |g|+ penalty which adds a slack variable t_g
        # for each constraint.
        # Each equality constraint is converted to a |h| penalty which adds slack variables t_h, s_h
        # for each constraint.
        # The A matrix then needs to be expanded (column wise) to account for the new variables
        # before adding new constraints to it as the new ones depend on the slack variables as well.

        num_nl_g_constraints, num_nl_h_constraints = 0, 0
        if self.non_linear_inequality_constraints_fn is not None:
            # The constraint for the non linear terms is slightly different as we have the slack terms.
            # Instead of Wx <= Wx0 - g(x0), we have
            # W@x - t_g <= W@x0 - g(x0)
            # As the actual constarint is g_linearized(x) <= t_g
            # Wx - t_g then can be represented using A@x where x now also includes the slack terms.
            # A = [[W 0]     x = [[x]
            #      [0 -I]]        [t_g]]
            nlg0 = self.non_linear_inequality_constraints_fn(x)
            W_nlg = self.non_linear_inequality_constraints_fn.grad(x)

            assert nlg0.ndim == 1
            num_nl_g_constraints = len(nlg0)

            # Expanding A to account for the new t_g slack variables. As the older constraints don't
            # depend on the slack variables, these can just be zero.
            A = np.hstack((A, np.zeros((A.shape[0], num_nl_g_constraints))))

            A_nlg = W_nlg
            ub_nlg = W_nlg @ x - nlg0
            # Lower limits are again -inf
            lb_nlg = np.full(len(ub_nlg), fill_value=-np.inf)

            # Expanding A_lh as well and changing values to account for the slack terms.
            # The slack terms will correspond to each constraint, so can be mapped using the identity matrix.
            assert A_nlg.shape[0] == num_nl_g_constraints
            A_nlg_aux = -1.0 * np.eye(num_nl_g_constraints, num_nl_g_constraints)
            A_nlg = np.hstack((A_nlg, A_nlg_aux))

            A = np.vstack((A, A_nlg))
            lb = np.hstack((lb, lb_nlg))
            ub = np.hstack((ub, ub_nlg))

        if self.non_linear_equality_constraints_fn is not None:
            nlh0 = self.non_linear_equality_constraints_fn(x)
            W_nlh = self.non_linear_equality_constraints_fn.grad(x)

            assert nlh0.ndim == 1
            num_nl_h_constraints = len(nlh0)

            # Doing the same thing, expanding the A matrices and accounting for the slack terms. It's slightly different
            # due to having two slack terms for each constraint row.

            A = np.hstack((A, np.zeros((A.shape[0], 2 * num_nl_h_constraints))))

            A_nlh = W_nlh
            b_nlh = W_nlh @ x - nlh0

            assert A_nlh.shape[0] == num_nl_h_constraints
            A_nlh_aux = np.zeros((num_nl_h_constraints, 2 * num_nl_h_constraints))
            for i in range(num_nl_h_constraints):
                # Constraint is W@x - t_h + s_h = W@x0 - h(x0)
                # Assuming the final x matrix is layed out as:
                # x = [[x]
                #      [t_h]
                #      [s_h]]
                A_nlh[i, i] = 1.0
                A_nlh[i, i + num_nl_h_constraints] = -1.0
            A_nlh = np.hstack((A_nlh, A_nlh_aux))

            A = np.vstack((A, A_nlh))
            lb = np.hstack((lb, b_nlh))
            ub = np.hstack((ub, b_nlh))

        num_slack_variables = num_nl_g_constraints + 2 * num_nl_h_constraints
        num_total_variables = n + num_slack_variables
        assert A.shape[1] == num_total_variables
        # Finally we add the trust region constraints as box inequalities.
        # x - s <= x <= x + s (We know that s >= 0.)
        lb_trust = x - s
        ub_trust = x + s

        A_trust = np.zeros((n, num_total_variables))
        A_trust[:n, :n] = np.eye(n)

        A = np.vstack((A, A_trust))
        lb = np.hstack((lb, lb_trust))
        ub = np.hstack((ub, ub_trust))

        # Computing the necessary gradients and hessians for the current x.
        # Cost function. Gradient vector by omega and hessian matrix by W
        omega_f = self.cost_fn.grad(x)
        W_f = self.cost_fn.hess(x)

        # For the quadratic term 0.5 x^T@P@x, P is just W_f expanded by zeros to account for the slack terms.
        P = np.zeros((num_total_variables, num_total_variables))
        P[:n, :n] = W_f

        # For the linear term q^Tx, the first part (x part) of q is given by
        # omega_f - 0.5 * (W_f + W_f^T)@x0
        # For proof: Expand f_convex(x) = f(x0) + omega_f^T@(x - x0) + 0.5 * (x - x0)^T@W_f@(x - x0)
        # f(x0) is not a function of x so can be ignored.
        q = omega_f - 0.5 * ((W_f + W_f.T) @ x)
        # The second part corresponds to the slack terms and are all equal to the penalty factor as
        # in the cost function they are sum(t_g) + sum(t_h + s_h)
        q_aux = np.full(num_slack_variables, fill_value=mu)
        q = np.hstack((q, q_aux))

        assert q.ndim == 1
        assert len(q) == num_total_variables

        return QPInputs(
            P=P,
            q=q,
            A=A,
            lb=lb,
            ub=ub,
        )

    def solve(
        self,
        initial_guess_x: VectorNf64,
    ) -> None:
        for penalty_iter in count():
            for convexify_iter in count():
                ...
            ...
            ...
