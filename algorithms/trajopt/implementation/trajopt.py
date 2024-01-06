"""
Implementation of the TrajOpt algorithm.

See: https://rll.berkeley.edu/~sachin/papers/Schulman-IJRR2014.pdf
"""
import time
from itertools import count
from typing import List, Optional

import attr
import numpy as np

from algorithms.trajopt.implementation.trajopt_utils import assert_gradient_sizes
from common.custom_types import VectorNf64
from common.exceptions import AtiumOptError
from common.math_utils import assert_matrix_positive_semidefinite
from common.optimization.constructs import QPInputs
from common.optimization.derivative_splicer import (
    DerivativeSplicedConstraintsFn,
    DerivativeSplicedCostFn,
)
from common.optimization.qp_solver import is_qp_solved, solve_qp


# x_tol <= s_0, tau_plus > 1, etc.
@attr.frozen
class TrajOptParams:
    # Optimization params.
    # Initial penalty coefficient
    mu_0: float = attr.ib(validator=attr.validators.ge(0.0))
    # Initial trust region size
    s_0: float = attr.ib(validator=attr.validators.gt(0.0))
    # Step acceptance parameter
    c: float = attr.ib(validator=attr.validators.gt(0.0))
    # Penalty scaling factor
    k: float = attr.ib(validator=attr.validators.gt(0.0))
    # Convergence thresholds
    f_tol: float = attr.ib(validator=attr.validators.gt(0.0))
    x_tol: float = attr.ib(validator=attr.validators.gt(0.0))
    # Constraint satisfaction threshold
    c_tol: float = attr.ib(validator=attr.validators.gt(0.0))

    # Trust region expansion and shrinkage factors
    tau_plus: float = attr.ib(validator=attr.validators.ge(1.0))
    tau_minus: float = attr.ib(validator=attr.validators.le(1.0))

    # Implementation params.
    tau_max: float = attr.ib()
    tau_min: float = attr.ib()
    max_iter: int
    # Whether or not to model the quadratic terms and approximate
    # the non linear equalities and inequalities as quadratic functions.
    # It is highly advised that this is done as non-linear constraint satisfaction
    # might prove difficult using just a linear approximation.
    second_order_inequalities: bool = True
    second_order_equalities: bool = True

    @tau_max.validator
    def _validate_tau_max(self, attribute, value) -> None:
        del attribute
        assert value >= self.s_0

    @tau_min.validator
    def _validate_tau_min(self, attribute, value) -> None:
        del attribute
        assert value >= self.x_tol


@attr.frozen
class TrajOptEntry:
    penalty_iter: int
    convexify_iter: int
    trust_region_iter: int
    min_x: VectorNf64
    updated_min_x: VectorNf64
    cost: float
    trust_region_size: float
    updated_trust_region_size: float
    improvement: bool
    trust_region_size_below_threshold: bool
    penalty_factor: float
    updated_penalty_factor: Optional[float] = None


@attr.define
class TrajOptResult:
    entries: List[TrajOptEntry] = attr.ib(factory=list)

    def __getitem__(self, key: int) -> TrajOptEntry:
        return self.entries[key]

    def __setitem__(self, key: int, value: TrajOptEntry):
        assert isinstance(value, TrajOptEntry)
        self.entries[key] = value

    def __len__(self) -> int:
        return len(self.entries)

    def record_entry(self, entry: TrajOptEntry) -> None:
        self.entries.append(entry)

    def solution_x(self) -> VectorNf64:
        return self[-1].updated_min_x if self[-1].improvement else self[-1].min_x


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
        mu: float,
    ) -> QPInputs:
        assert x.ndim == 1
        n = x.size
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

        # W stores the accumulation of the required hessians (second order terms) in the cost function.
        W = np.zeros((n, n), dtype=np.float64)
        # q stores the accumulation of the linear cost function terms.
        q = np.zeros(n, dtype=np.float64)
        # A, lb and ub stores the accumulation of the linear (and linearized/linear parts of) constraints
        # in the form lb <= Ax <= ub as this is what OSQP requires as input.
        A = np.empty((0, n), dtype=np.float64)
        lb, ub = np.empty(0), np.empty(0)

        if self.linear_inequality_constraints_fn is not None:
            lg0 = self.linear_inequality_constraints_fn(x)
            W_lg = self.linear_inequality_constraints_fn.grad(x)

            assert_gradient_sizes(
                x0=lg0,
                x_grad=W_lg,
                num_variables=n,
            )

            A_lg = W_lg
            ub_lg = W_lg @ x - lg0
            # Lower limits are all -inf for this set.
            lb_lg = np.full(ub_lg.size, fill_value=-np.inf)

            A = np.vstack((A, A_lg))
            lb = np.hstack((lb, lb_lg))
            ub = np.hstack((ub, ub_lg))

        if self.linear_equality_constraints_fn is not None:
            lh0 = self.linear_equality_constraints_fn(x)
            W_lh = self.linear_equality_constraints_fn.grad(x)

            assert_gradient_sizes(
                x0=lh0,
                x_grad=W_lh,
                num_variables=n,
            )

            A_lh = W_lh
            b_lh = W_lh @ x - lh0

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
            num_nl_g_constraints = nlg0.size

            assert_gradient_sizes(
                x0=nlg0,
                x_grad=W_nlg,
                num_variables=n,
            )

            # Expanding A to account for the new t_g slack variables. As the older constraints don't
            # depend on the slack variables, these can just be zero.
            A = np.hstack(
                (A, np.zeros((A.shape[0], num_nl_g_constraints), dtype=np.float64))
            )

            A_nlg = W_nlg
            ub_nlg = W_nlg @ x - nlg0
            # Lower limits are again -inf
            lb_nlg = np.full(ub_nlg.size, fill_value=-np.inf)

            # Expanding A_lh as well and changing values to account for the slack terms.
            # The slack terms will correspond to each constraint, so can be mapped using the identity matrix.
            A_nlg_aux = -1.0 * np.eye(num_nl_g_constraints)
            if num_nl_g_constraints == 1:
                A_nlg = np.hstack((A_nlg, A_nlg_aux.squeeze()))
            else:
                A_nlg = np.hstack((A_nlg, A_nlg_aux))

            A = np.vstack((A, A_nlg))
            lb = np.hstack((lb, lb_nlg))
            ub = np.hstack((ub, ub_nlg))

            if self.params.second_order_inequalities:
                # If we're required to model the inequalities as quadratic terms, the hessian term goes in the
                # cost function and the linear term goes into the constraints directly.
                # This is because |g(x)|+ = |g(x0) + W(x0)@(x - x0) + Sum_i (x - x0)^T@Omega[i]@(x - x0)|
                # Where |g(x)|+ = max(g(x), 0) and Omega = Hessian tensor (matrix for a single constraint)
                # If the individual Omega[:, :, i] are positive semi-definite, then we can remove this out of the
                # max() as dx^T @ Omega[i] @ dx >= 0.
                # Which then gets added to the cost function as:
                # |g(x)|+ = sum_i dx^T @ Omega[i] @ dx + t_g
                # Ax + b <= t_g, t_g >= 0, Where Ax + b is the linear form of the approximation.

                # So here for the cost, we expand dx^T @ Omega[i] @ dx and accumulate the quadratic terms
                # (0.5 * Omega[i]) in W and the linear terms (-0.5 * x0^T @ (Omega[i] + Omega[i]^T)) into q.
                # Note that the penalty scaling factor mu also multiplies the cost function term.

                Omega_nlg = self.non_linear_inequality_constraints_fn.hess(x)

                if num_nl_g_constraints == 1:
                    # Omega is not a tensor in this case.
                    assert_matrix_positive_semidefinite(mat=Omega_nlg)
                    # Note that OSQP already assumes 0.5 multiplies P, so we don't include that here.
                    W += mu * Omega_nlg
                    # For q, we need to include the 0.5
                    # Note that v.T @ (A + A.T) = ((A + A.T).T @ v).T = (A + A.T) @ v (As it's a vector and the transpose doesn't change the values)
                    q -= 0.5 * mu * ((Omega_nlg + Omega_nlg.T) @ x)

                else:
                    assert Omega_nlg.ndim == 3
                    assert Omega_nlg.shape == (n, n, num_nl_g_constraints)
                    for i in range(n):
                        omega_nlg = Omega_nlg[:, :, i]
                        assert_matrix_positive_semidefinite(mat=omega_nlg)
                        W += mu * omega_nlg
                        q -= 0.5 * mu * ((omega_nlg + omega_nlg.T) @ x)

        if self.non_linear_equality_constraints_fn is not None:
            nlh0 = self.non_linear_equality_constraints_fn(x)
            W_nlh = self.non_linear_equality_constraints_fn.grad(x)
            num_nl_h_constraints = nlh0.size

            assert_gradient_sizes(
                x0=nlh0,
                x_grad=W_nlh,
                num_variables=n,
            )

            # Doing the same thing, expanding the A matrices and accounting for the slack terms. It's slightly different
            # due to having two slack terms for each constraint row.

            A = np.hstack(
                (A, np.zeros((A.shape[0], 2 * num_nl_h_constraints), dtype=np.float64))
            )
            # Offset due to any previous nlg constraints.
            nlg_offset = A.shape[1] - n - 2 * num_nl_h_constraints

            A_nlh = W_nlh
            b_nlh = W_nlh @ x - nlh0

            A_nlh_aux = np.zeros(
                (num_nl_h_constraints, A.shape[1] - n), dtype=np.float64
            )
            for i in range(num_nl_h_constraints):
                # Constraint is W@x - t_h + s_h = W@x0 - h(x0)
                # Assuming the final x matrix is layed out as:
                # x = [[x]
                #      [t_h]
                #      [s_h]]
                # Also we need to add the offset to make sure we ignore the g slack terms
                # if any. If no nlg constraints, this offset will be = 0.
                A_nlh_aux[i, nlg_offset + i] = 1.0
                A_nlh_aux[i, nlg_offset + i + num_nl_h_constraints] = -1.0

            if num_nl_h_constraints == 1:
                # A_lh is 1-D and A_nlh_aux is a 2-D but in 1-D form
                A_nlh_aux = A_nlh_aux.reshape(nlg_offset + 2 * num_nl_h_constraints)
            A_nlh = np.hstack((A_nlh, A_nlh_aux))

            A = np.vstack((A, A_nlh))
            lb = np.hstack((lb, b_nlh))
            ub = np.hstack((ub, b_nlh))

            # Doing the same thing for the equality constraints if required.
            if self.params.second_order_equalities:
                Omega_nlh = self.non_linear_equality_constraints_fn.hess(x)

                if num_nl_h_constraints == 1:
                    # Omega is not a tensor in this case.
                    assert_matrix_positive_semidefinite(mat=Omega_nlh)
                    # Note that OSQP already assumes 0.5 multiplies P, so we don't include that here.
                    W += mu * Omega_nlh
                    # For q, we need to include the 0.5
                    q -= 0.5 * mu * ((Omega_nlh + Omega_nlh.T) @ x)

                else:
                    assert Omega_nlh.ndim == 3
                    assert Omega_nlh.shape == (n, n, num_nl_h_constraints)
                    for i in range(n):
                        omega_nlh = Omega_nlh[:, :, i]
                        assert_matrix_positive_semidefinite(mat=omega_nlh)
                        W += mu * omega_nlh
                        q -= 0.5 * mu * ((omega_nlh + omega_nlh.T) @ x)

        num_slack_variables = num_nl_g_constraints + 2 * num_nl_h_constraints
        num_total_variables = n + num_slack_variables
        assert A.shape[1] == num_total_variables

        # Constraints for the slack terms t_g, t_h and s_h to be >= 0
        A_slack_bounds = np.zeros(
            (num_slack_variables, num_total_variables), dtype=np.float64
        )
        if num_slack_variables:
            A_slack_bounds[-num_slack_variables:, -num_slack_variables:] = np.eye(
                num_slack_variables
            )
        lb_slack = np.zeros(num_slack_variables)
        ub_slack = np.full(num_slack_variables, np.inf)

        A = np.vstack((A, A_slack_bounds))
        lb = np.hstack((lb, lb_slack))
        ub = np.hstack((ub, ub_slack))

        # Computing the necessary gradients and hessians for the current x.
        # Cost function. Gradient vector by omega and hessian matrix by W
        omega_f = self.cost_fn.grad(x)
        W_f = self.cost_fn.hess(x)

        # For the quadratic term 0.5 x^T@P@x, P is just W_f expanded by zeros to account for the slack terms.
        P = np.zeros((num_total_variables, num_total_variables), dtype=np.float64)
        P[:n, :n] = W + W_f
        # Making sure that it is positive semi-definite.
        assert_matrix_positive_semidefinite(mat=P)

        # For the linear term q^Tx, the first part (x part) of q is given by
        # omega_f - 0.5 * (W_f + W_f^T)@x0
        # For proof: Expand f_convex(x) = f(x0) + omega_f^T@(x - x0) + 0.5 * (x - x0)^T@W_f@(x - x0)
        # f(x0) is not a function of x so can be ignored.
        q += omega_f - 0.5 * ((W_f + W_f.T) @ x)
        # The second part corresponds to the slack terms and are all equal to the penalty factor as
        # in the cost function they are sum(t_g) + sum(t_h + s_h)
        q_aux = np.full(num_slack_variables, fill_value=mu)
        q = np.hstack((q, q_aux))

        assert q.ndim == 1
        assert q.size == num_total_variables

        return QPInputs(
            P=P,
            q=q,
            A=A,
            lb=lb,
            ub=ub,
        )

    def incorporate_trust_region(
        self,
        x: VectorNf64,
        s: float,
        qp_inputs: QPInputs,
    ) -> QPInputs:
        assert x.ndim == 1
        assert s > 0.0, f"s: {s} is not > 0."
        # Adding the trust region constraints as box inequalities.
        # x - s <= x <= x + s (We know that s >= 0.)
        lb_trust = x - s
        ub_trust = x + s

        n = x.size
        num_total_variables = qp_inputs.A.shape[1]

        A_trust_bounds = np.zeros((n, num_total_variables), dtype=np.float64)
        A_trust_bounds[:n, :n] = np.eye(n)

        A = np.vstack((qp_inputs.A, A_trust_bounds))
        lb = np.hstack((qp_inputs.lb, lb_trust))
        ub = np.hstack((qp_inputs.ub, ub_trust))

        return attr.evolve(
            qp_inputs,
            A=A,
            lb=lb,
            ub=ub,
        )

    def compute_convexified_x(self, qp_inputs: QPInputs, size_x: int) -> VectorNf64:
        osqp_results = solve_qp(qp_inputs=qp_inputs, verbose=False)
        if is_qp_solved(osqp_results=osqp_results):
            return osqp_results.x[:size_x]
        else:
            raise AtiumOptError(
                f"QP could not be solved using OSQP. Status is: {osqp_results.info.status}"
            )

    def is_improvement(
        self,
        x: VectorNf64,
        new_x: VectorNf64,
    ) -> bool:
        # new_x should have a lower cost, so improvement is f(old_x) - f(new_x)
        true_improve = self.cost_fn(x) - self.cost_fn(new_x)
        # For the model improvement, we measure the difference between the cost at x (previous)
        # and the convexified cost at new_x. The convexified cost at x is basically just
        # the full cost at x as delta_x is zero
        # model_improve = self.cost_fn(x) - self.convexified_cost_fn(x=x, new_x=new_x)
        model_improve = self.cost_fn(x) - self.cost_fn.convexified(x=x, new_x=new_x)

        return true_improve / model_improve > self.params.c

    def is_converged(
        self,
        x: VectorNf64,
        new_x: VectorNf64,
    ) -> bool:
        x_converged = np.linalg.norm(new_x - x) < self.params.x_tol
        f_converged = (
            np.linalg.norm(self.cost_fn(new_x) - self.cost_fn(x)) < self.params.f_tol
        )

        return x_converged or f_converged

    def are_constraints_satisfied(
        self,
        x: VectorNf64,
    ) -> bool:
        constraints_satisfied = True
        if self.linear_inequality_constraints_fn is not None:
            lg_satisfied = np.all(
                self.linear_inequality_constraints_fn(x) <= self.params.c_tol
            )
            constraints_satisfied = constraints_satisfied and lg_satisfied
        if self.linear_equality_constraints_fn is not None:
            lh_satisfied = np.allclose(
                self.linear_equality_constraints_fn(x),
                0.0,
                atol=self.params.c_tol,
            )
            constraints_satisfied = constraints_satisfied and lh_satisfied
        if self.non_linear_inequality_constraints_fn is not None:
            nlg_satisfied = np.all(
                self.non_linear_inequality_constraints_fn(x) <= self.params.c_tol
            )
            constraints_satisfied = constraints_satisfied and nlg_satisfied
        if self.non_linear_equality_constraints_fn is not None:
            nlh_satisfied = np.allclose(
                self.non_linear_equality_constraints_fn(x),
                0.0,
                atol=self.params.c_tol,
            )
            constraints_satisfied = constraints_satisfied and nlh_satisfied
        return constraints_satisfied

    def solve(
        self,
        initial_guess_x: VectorNf64,
    ) -> TrajOptResult:
        trajopt_solve_start_time = time.perf_counter()
        # Initial values of variables and optimization params.
        x = initial_guess_x
        s = self.params.s_0
        mu = self.params.mu_0

        size_x = len(initial_guess_x)
        new_x = np.copy(x)
        updated_s = s
        improvement = True

        result = TrajOptResult()
        # Add initial entry for the initial state and x
        result.record_entry(
            entry=TrajOptEntry(
                penalty_iter=0,
                convexify_iter=0,
                trust_region_iter=0,
                min_x=initial_guess_x,
                updated_min_x=initial_guess_x,
                cost=self.cost_fn(initial_guess_x),
                trust_region_size=s,
                updated_trust_region_size=s,
                improvement=improvement,
                trust_region_size_below_threshold=False,
                penalty_factor=mu,
            )
        )

        for penalty_iter in range(self.params.max_iter):
            for convexify_iter in count():
                trust_region_size_below_threshold = False
                qp_inputs = self.convexify_problem(
                    x=x,
                    mu=mu,
                )
                for trust_region_iter in count():
                    if improvement:
                        x = new_x
                    s = updated_s
                    qp_inputs = self.incorporate_trust_region(
                        x=x,
                        s=s,
                        qp_inputs=qp_inputs,
                    )
                    # Solving the QP
                    new_x = self.compute_convexified_x(
                        qp_inputs=qp_inputs,
                        size_x=size_x,
                    )
                    # input()
                    cost = self.cost_fn(new_x)
                    improvement = self.is_improvement(x=x, new_x=new_x)

                    result.record_entry(
                        entry=TrajOptEntry(
                            penalty_iter=penalty_iter,
                            convexify_iter=convexify_iter,
                            trust_region_iter=trust_region_iter,
                            min_x=x,
                            updated_min_x=new_x,
                            cost=cost,
                            trust_region_size=s,
                            updated_trust_region_size=updated_s,
                            improvement=improvement,
                            trust_region_size_below_threshold=trust_region_size_below_threshold,
                            penalty_factor=mu,
                        ),
                    )

                    if improvement:
                        updated_s = min(self.params.tau_plus * s, self.params.tau_max)
                        updated_s = max(updated_s, 1.0)
                        break
                    else:
                        updated_s = max(self.params.tau_minus * s, self.params.tau_min)
                    if updated_s < self.params.x_tol:
                        trust_region_size_below_threshold = True
                        break

                if trust_region_size_below_threshold or self.is_converged(
                    x=x, new_x=new_x
                ):
                    break
            if self.are_constraints_satisfied(x=new_x):
                # TODO: Log
                print(
                    f"TrajOpt found a solution in {time.perf_counter() - trajopt_solve_start_time} seconds!"
                )
                print(f"Optimal x: {result.solution_x()}")
                break
            else:
                # mu = min(self.params.k * mu, 1e10)
                mu = self.params.k * mu
                # Adding the updated penalty in the result.
                result[-1] = attr.evolve(
                    result[-1],
                    updated_penalty_factor=mu,
                )
        else:
            # TODO: Log
            print(
                f"TrajOpt failed to find a solution within {self.params.max_iter} iterations!"
            )

        return result
