import numpy as np
from pydrake.autodiffutils import AutoDiffXd

from atium.implementations.unito.src.unito_variable_manager import UnitoVariableManager


def control_cost_func(
    all_vars: np.ndarray,
    manager: UnitoVariableManager,
) -> float | AutoDiffXd:
    cost = 0.0
    # sigma(i, n-1) = sigma(i + 1, 0)
    # So we only go up to n-2  so that we don't double count.
    for i in range(manager.params.M):
        for j in range(manager.params.n - 1):
            t_ij_exp = manager.get_t_ij_exp(
                t_vars=manager.get_t_vars(all_vars),
                i=i,
                j=j,
            )
            sigma_i = manager.get_sigma_ij_exp(
                c_theta_i_vars=manager.get_c_theta_i_vars(all_vars, i),
                c_s_i_vars=manager.get_c_s_i_vars(all_vars, i),
                t_ij_exp=t_ij_exp,
                derivative=manager.params.h,
            )
            cost += sigma_i @ manager.params.W @ sigma_i.T
            print(cost)
    return cost


def time_regularization_cost_func(
    t_vars: np.ndarray,
    manager: UnitoVariableManager,
) -> float | AutoDiffXd:
    """
    Cost on the time taken to traverse all segments.
    Returns epsilon_t * sum(t_i)
    """
    return manager.params.epsilon_t * np.sum(t_vars)
