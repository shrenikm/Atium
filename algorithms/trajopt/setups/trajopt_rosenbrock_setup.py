from typing import Optional

import attr
import jax.numpy as jnp

from algorithms.trajopt.implementation.trajopt import TrajOpt, TrajOptParams
from common.custom_types import VectorNf64
from common.optimization.derivative_splicer import (
    DerivativeSplicedConstraintsFn,
    DerivativeSplicedCostFn,
)
from common.optimization.standard_functions.rosenbrock import (
    RosenbrockParams,
    rosenbrock_cost_fn,
)


@attr.frozen
class RosenbrockOptParamsConstructor:
    params: RosenbrockParams

    def __call__(self, x: VectorNf64) -> RosenbrockParams:
        return self.params


def lg_fn1(z: VectorNf64) -> VectorNf64:
    """
    Linear inequality constraints of the form:
    x >= c, y >= d
    (x, y) >= (c, d) => (x-c, y-d) >= 0 => (c-x, d-y) <= 0
    """
    x, y = z
    return jnp.array([2.0 - x, -2.0 - y], dtype=jnp.float32)


def nlg_fn1(z: VectorNf64) -> VectorNf64:
    """
    Non linear inequality (circle) constraints of the form:
    (x - xc)^2 + (y - yc)^2 - r^2 <= 0
    """
    x, y = z
    xc, yc = (2.0, 2.0)
    r = 2.0
    return jnp.array((x - xc) ** 2 + (y - yc) ** 2 - r**2, dtype=jnp.float32)


def nlh_fn1(z: VectorNf64) -> VectorNf64:
    """
    Non linear equality (circle) constraints of the form:
    (x - xc)^2 + (y - yc)^2 - r^2 == 0
    """
    x, y = z
    xc, yc = (2.0, 2.0)
    r = 1.5
    return jnp.array((x - xc) ** 2 + (y - yc) ** 2 - r**2, dtype=jnp.float32)


def setup_trajopt_for_rosenbrock(
    rosenbrock_params: RosenbrockParams,
    trajopt_params: Optional[TrajOptParams] = None,
) -> TrajOpt:
    if trajopt_params is None:
        # Default trajopt params if not given.
        # TODO: Setup somewhere if not too problem specific.
        trajopt_params = TrajOptParams(
            mu_0=1.0,
            s_0=1e-4,
            c=1e-2,
            k=10.0,
            f_tol=1e-6,
            x_tol=1e-6,
            c_tol=1e-4,
            tau_plus=1.5,
            tau_minus=0.1,
            tau_max=10.0,
            tau_min=1e-5,
            max_iter=200,
            second_order_inequalities=True,
            second_order_equalities=True,
        )

    rosenbrock_params = RosenbrockParams(a=1.0, b=100.0)
    rosenbrock_params_constructor = RosenbrockOptParamsConstructor(
        params=rosenbrock_params
    )

    cost_fn_ds = DerivativeSplicedCostFn(
        core_fn=rosenbrock_cost_fn,
        use_jit=True,
        construct_params_fn=rosenbrock_params_constructor,
    )

    lg_fn_ds = DerivativeSplicedConstraintsFn(
        core_fn=lg_fn1,
        use_jit=True,
    )
    nlg_fn_ds = DerivativeSplicedConstraintsFn(
        core_fn=nlg_fn1,
        use_jit=True,
    )
    nlh_fn_ds = DerivativeSplicedConstraintsFn(
        core_fn=nlh_fn1,
        use_jit=True,
    )

    trajopt_optimizer = TrajOpt(
        params=trajopt_params,
        cost_fn=cost_fn_ds,
        linear_inequality_constraints_fn=lg_fn_ds,
        non_linear_inequality_constraints_fn=nlg_fn_ds,
        non_linear_equality_constraints_fn=nlh_fn_ds,
    )

    return trajopt_optimizer
