from typing import Optional

import attr

from algorithms.trajopt.implementation.trajopt import TrajOpt, TrajOptParams
from common.custom_types import VectorNf64
from common.optimization.derivative_splicer import DerivativeSplicedCostFn
from common.optimization.standard_functions.rosenbrock import (
    RosenbrockParams,
    rosenbrock_cost_fn,
)


@attr.frozen
class RosenbrockOptParamsConstructor:
    params: RosenbrockParams

    def __call__(self, x: VectorNf64) -> RosenbrockParams:
        return self.params


def setup_trajopt_for_rosenbrock(
    rosenbrock_params: RosenbrockParams,
    trajopt_params: Optional[TrajOptParams] = None,
) -> TrajOpt:

    if trajopt_params is None:
        # Default trajopt params if not given.
        # TODO: Setup somewhere if not too problem specific.
        trajopt_params = TrajOptParams(
            mu_0=0.01,
            s_0=0.01,
            c=1e-4,
            tau_plus=10.0,
            tau_minus=0.1,
            k=10.0,
            f_tol=1e-4,
            x_tol=1e-4,
            c_tol=1e-4,
            max_iter=1000,
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

    trajopt_optimizer = TrajOpt(
        params=trajopt_params,
        cost_fn=cost_fn_ds,
    )

    return trajopt_optimizer
