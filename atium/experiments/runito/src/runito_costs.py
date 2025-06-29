import numpy as np
from pydrake.autodiffutils import AutoDiffXd

from atium.experiments.runito.src.runito_variable_manager import RunitoVariableManager


def control_cost_func(
    all_vars: np.ndarray,
    manager: RunitoVariableManager,
) -> float | AutoDiffXd:
    cost = 0.0
    for i in range(manager.params.M):
        t_i_var = manager.get_t_i_var(all_vars, i=i)
        sigma_i = manager.compute_sigma_i_exp(
            c_x_i_vars=manager.get_c_x_i_vars(all_vars, i),
            c_y_i_vars=manager.get_c_y_i_vars(all_vars, i),
            c_theta_i_vars=manager.get_c_theta_i_vars(all_vars, i),
            t_exp=t_i_var,
            derivative=manager.params.h,
        )
        cost += sigma_i @ manager.params.W @ sigma_i.T
    return cost


def time_regularization_cost_func(
    t_vars: np.ndarray,
    manager: RunitoVariableManager,
) -> float | AutoDiffXd:
    """
    Cost on the time taken to traverse all segments.
    Returns epsilon_t * sum(t_i)
    """
    return manager.params.epsilon_t * np.sum(t_vars)
