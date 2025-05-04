import numpy as np

from atium.core.constructs.environment_map import EnvironmentMap2D
from atium.core.utils.custom_types import MatrixMNf32, PolygonXYArray, PositionXYVector, StateVector
from atium.implementations.unito.src.unito_variable_manager import UnitoVariableManager


def initial_ms_constraint_func(
    func_vars: np.ndarray,
    initial_ms_vector: StateVector,
    derivative: int,
    manager: UnitoVariableManager,
) -> np.ndarray:
    assert func_vars.shape == (4 * manager.params.h,)

    c_theta_0_vars = func_vars[: 2 * manager.params.h]
    c_s_0_vars = func_vars[2 * manager.params.h :]

    sigma_0 = manager.compute_sigma_i_exp(
        c_theta_i_vars=c_theta_0_vars,
        c_s_i_vars=c_s_0_vars,
        t_exp=0.0,
        derivative=derivative,
    )
    return sigma_0 - initial_ms_vector


def final_ms_constraint_func(
    func_vars: np.ndarray,
    final_ms_vector: StateVector,
    derivative: int,
    manager: UnitoVariableManager,
) -> np.ndarray:
    assert func_vars.shape == (4 * manager.params.h + 1,)

    c_f_vars = func_vars[:-1]
    t_f_var = func_vars[-1]

    assert c_f_vars.shape == (4 * manager.params.h,)

    c_theta_f_vars = c_f_vars[: 2 * manager.params.h]
    c_s_f_vars = c_f_vars[2 * manager.params.h :]

    sigma_f = manager.compute_sigma_i_exp(
        c_theta_i_vars=c_theta_f_vars,
        c_s_i_vars=c_s_f_vars,
        t_exp=t_f_var,
        derivative=derivative,
    )
    return sigma_f - final_ms_vector


def continuity_constraint_func(
    func_vars: np.ndarray,
    derivative: int,
    manager: UnitoVariableManager,
) -> np.ndarray:
    assert func_vars.shape == (4 * 2 * manager.params.h + 1,)

    prev_c_vars = func_vars[: 4 * manager.params.h]
    next_c_vars = func_vars[4 * manager.params.h : 4 * 2 * manager.params.h]
    prev_t_var = func_vars[-1]
    prev_c_theta_vars = prev_c_vars[: 2 * manager.params.h]
    prev_c_s_vars = prev_c_vars[2 * manager.params.h :]
    next_c_theta_vars = next_c_vars[: 2 * manager.params.h]
    next_c_s_vars = next_c_vars[2 * manager.params.h :]

    prev_sigma = manager.compute_sigma_i_exp(
        c_theta_i_vars=prev_c_theta_vars,
        c_s_i_vars=prev_c_s_vars,
        t_exp=prev_t_var,
        derivative=derivative,
    )
    next_sigma = manager.compute_sigma_i_exp(
        c_theta_i_vars=next_c_theta_vars,
        c_s_i_vars=next_c_s_vars,
        t_exp=0.0,
        derivative=derivative,
    )
    return prev_sigma - next_sigma


def final_xy_constraint_func(
    func_vars: np.ndarray,
    final_xy: PositionXYVector,
    initial_xy: PositionXYVector,
    manager: UnitoVariableManager,
) -> np.ndarray:
    assert func_vars.shape == (4 * manager.params.h * manager.params.M + manager.params.M,)

    xf = manager.compute_x_ij_exp(
        all_vars=func_vars,
        initial_x=initial_xy[0],
        i=manager.params.M - 1,
        j=manager.params.n - 1,
    )
    yf = manager.compute_y_ij_exp(
        all_vars=func_vars,
        initial_y=initial_xy[1],
        i=manager.params.M - 1,
        j=manager.params.n - 1,
    )

    return np.array([xf - final_xy[0], yf - final_xy[1]])


def obstacle_constraint_func(
    func_vars: np.ndarray,
    footprint: PolygonXYArray,
    emap2d: EnvironmentMap2D,
    # Signed distance can be computed from the emap, but we have a separate argument
    # in order to avoid recomputing it each time the constraint is evaluated.
    signed_distance_map: MatrixMNf32,
    obstacle_clearance: float,
    initial_xy: PositionXYVector,
    manager: UnitoVariableManager,
) -> np.ndarray:
    assert func_vars.shape == (4 * manager.params.h * manager.params.M + manager.params.M,)

    constraint_vector = []
    h, w = signed_distance_map.shape

    for i in range(manager.params.M):
        for j in range(manager.params.n):
            x_ij = manager.compute_x_ij_exp(
                all_vars=func_vars,
                initial_x=initial_xy[0],
                i=i,
                j=j,
            )
            y_ij = manager.compute_y_ij_exp(
                all_vars=func_vars,
                initial_y=initial_xy[1],
                i=i,
                j=j,
            )
            theta_ij = manager.compute_t_ijl_exp(
                t_i_var=manager.get_t_i_var(func_vars, i),
                j=j,
                l=0,
            )
            # Need to spell this transform out so that we get gradient information.
            # TODO: Support Autodiff in transform utils.
            rotation_matrix = np.array(
                [
                    [np.cos(theta_ij), -np.sin(theta_ij)],
                    [np.sin(theta_ij), np.cos(theta_ij)],
                ]
            )
            transformed_footprint = rotation_matrix @ footprint.T + np.array([[x_ij], [y_ij]])
            transformed_footprint = transformed_footprint.T

            for footprint_i in range(transformed_footprint.shape[0]):
                # Bilinear interpolation.
                fx = transformed_footprint[footprint_i][0]
                fy = transformed_footprint[footprint_i][1]

                px_x, px_y = emap2d.xy_to_px(
                    xy=(fx, fy),
                    output_as_float=True,
                )

                px_x0, px_y0 = int(np.floor(px_x)), int(np.floor(px_y))
                px_x1, px_y1 = px_x0 + 1, px_y0 + 1

                # TODO: Maybe make this an emap2d function.
                px_x0 = np.clip(px_x0, 0, w - 1)
                px_x1 = np.clip(px_x1, 0, w - 1)
                px_y0 = np.clip(px_y0, 0, h - 1)
                px_y1 = np.clip(px_y1, 0, h - 1)

                sd1 = signed_distance_map[px_y0, px_x0]
                sd2 = signed_distance_map[px_y0, px_x1]
                sd3 = signed_distance_map[px_y1, px_x0]
                sd4 = signed_distance_map[px_y1, px_x1]

                dx = px_x - px_x0
                dy = px_y - px_y0

                sd5 = sd1 * (1 - dx) + sd2 * dx
                sd6 = sd3 * (1 - dx) + sd4 * dx
                sd = sd5 * (1 - dy) + sd6 * dy

                constraint_vector.append(sd - obstacle_clearance)

    return np.array(constraint_vector)
