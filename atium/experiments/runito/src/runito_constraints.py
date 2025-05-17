import numpy as np

from atium.core.definitions.concrete_states import Pose2D, Velocity2D
from atium.core.definitions.environment_map import EnvironmentMap2D
from atium.core.utils.custom_types import MatrixMNf32, PolygonXYArray, Pose2DVector
from atium.core.utils.geometry_utils import normalize_angle, normalize_angle_differentiable
from atium.experiments.runito.src.runito_variable_manager import RunitoVariableManager


def _compute_pose_diff(
    pose_vector: Pose2DVector,
    expected_pose_vector: Pose2DVector,
) -> np.ndarray:
    """
    Compute the difference between the pose vector (variable)
    and the expected pose (construct).
    """
    pose_diff = pose_vector - expected_pose_vector
    pose_diff[2] = normalize_angle_differentiable(pose_diff[2])
    return pose_diff


def initial_pose_constraint_func(
    func_vars: np.ndarray,
    initial_pose: Pose2D,
    manager: RunitoVariableManager,
) -> np.ndarray:
    assert func_vars.shape == (3 * manager.num_ci,)

    c_x_0_vars = func_vars[: manager.num_ci]
    c_y_0_vars = func_vars[manager.num_ci : 2 * manager.num_ci]
    c_theta_0_vars = func_vars[2 * manager.num_ci : 3 * manager.num_ci]

    sigma_0 = manager.compute_sigma_i_exp(
        c_x_i_vars=c_x_0_vars,
        c_y_i_vars=c_y_0_vars,
        c_theta_i_vars=c_theta_0_vars,
        t_exp=0.0,
    )
    return _compute_pose_diff(
        pose_vector=sigma_0,
        expected_pose_vector=initial_pose.to_vector(),
    )


def initial_velocity_constraint_func(
    func_vars: np.ndarray,
    initial_velocity: Velocity2D,
    manager: RunitoVariableManager,
) -> np.ndarray:
    assert func_vars.shape == (6 * manager.params.h,)

    c_x_0_vars = func_vars[: manager.num_ci]
    c_y_0_vars = func_vars[manager.num_ci : 2 * manager.num_ci]
    c_theta_0_vars = func_vars[2 * manager.num_ci : 3 * manager.num_ci]

    gamma_0 = manager.compute_gamma_i_exp(
        c_x_i_vars=c_x_0_vars,
        c_y_i_vars=c_y_0_vars,
        c_theta_i_vars=c_theta_0_vars,
        t_exp=0.0,
    )
    return gamma_0 - initial_velocity.to_vector()


def final_pose_constraint_func(
    func_vars: np.ndarray,
    final_pose: Pose2D,
    manager: RunitoVariableManager,
) -> np.ndarray:
    assert func_vars.shape == (3 * manager.num_ci + 1,)

    c_f_vars = func_vars[:-1]
    t_f_var = func_vars[-1]

    c_x_f_vars = c_f_vars[: manager.num_ci]
    c_y_f_vars = c_f_vars[manager.num_ci : 2 * manager.num_ci]
    c_theta_f_vars = c_f_vars[2 * manager.num_ci : 3 * manager.num_ci]

    sigma_f = manager.compute_sigma_i_exp(
        c_x_i_vars=c_x_f_vars,
        c_y_i_vars=c_y_f_vars,
        c_theta_i_vars=c_theta_f_vars,
        t_exp=t_f_var,
    )
    return _compute_pose_diff(
        pose_vector=sigma_f,
        expected_pose_vector=final_pose.to_vector(),
    )


def final_velocity_constraint_func(
    func_vars: np.ndarray,
    final_velocity: Velocity2D,
    manager: RunitoVariableManager,
) -> np.ndarray:
    assert func_vars.shape == (3 * manager.num_ci + 1,)

    c_f_vars = func_vars[:-1]
    t_f_var = func_vars[-1]

    c_x_f_vars = c_f_vars[: manager.num_ci]
    c_y_f_vars = c_f_vars[manager.num_ci : 2 * manager.num_ci]
    c_theta_f_vars = c_f_vars[2 * manager.num_ci : 3 * manager.num_ci]

    gamma_f = manager.compute_gamma_i_exp(
        c_x_i_vars=c_x_f_vars,
        c_y_i_vars=c_y_f_vars,
        c_theta_i_vars=c_theta_f_vars,
        t_exp=t_f_var,
    )
    return gamma_f - final_velocity.to_vector()


def velocity_limits_constraint_func(
    func_vars: np.ndarray,
    manager: RunitoVariableManager,
    only_at_segment_endpoints: bool,
) -> np.ndarray:
    assert func_vars.shape == (3 * manager.num_c + manager.params.M,)

    linear_constraint_vector = []
    angular_constraint_vector = []

    for i in range(manager.params.M):
        c_x_i_vars = manager.get_c_x_i_vars(all_vars=func_vars, i=i)
        c_y_i_vars = manager.get_c_y_i_vars(all_vars=func_vars, i=i)
        c_theta_i_vars = manager.get_c_theta_i_vars(all_vars=func_vars, i=i)
        t_i_var = manager.get_t_i_var(func_vars, i)

        if only_at_segment_endpoints:
            t_ijl = manager.compute_t_ijl_exp(
                t_i_var=t_i_var,
                j=manager.params.n - 1,
                l=0,
            )

            gamma_i = manager.compute_gamma_i_exp(
                c_x_i_vars=c_x_i_vars,
                c_y_i_vars=c_y_i_vars,
                c_theta_i_vars=c_theta_i_vars,
                t_exp=t_ijl,
            )

            linear_constraint_vector.append(gamma_i[0])
            angular_constraint_vector.append(gamma_i[1])
        else:
            for j in range(manager.params.n):
                t_ijl = manager.compute_t_ijl_exp(
                    t_i_var=t_i_var,
                    j=j,
                    l=0,
                )

                gamma_i = manager.compute_gamma_i_exp(
                    c_x_i_vars=c_x_i_vars,
                    c_y_i_vars=c_y_i_vars,
                    c_theta_i_vars=c_theta_i_vars,
                    t_exp=t_ijl,
                )

                linear_constraint_vector.append(gamma_i[0])
                angular_constraint_vector.append(gamma_i[1])

    return np.hstack((linear_constraint_vector, angular_constraint_vector))


def continuity_constraint_func(
    func_vars: np.ndarray,
    derivative: int,
    manager: RunitoVariableManager,
) -> np.ndarray:
    assert func_vars.shape == (6 * manager.num_ci + 1,)

    prev_c_vars = func_vars[: 3 * manager.num_ci]
    next_c_vars = func_vars[3 * manager.num_ci : 6 * manager.num_ci]
    prev_c_x_vars = prev_c_vars[: manager.num_ci]
    prev_c_y_vars = prev_c_vars[manager.num_ci : 2 * manager.num_ci]
    prev_c_theta_vars = prev_c_vars[2 * manager.num_ci : 3 * manager.num_ci]
    next_c_x_vars = next_c_vars[: manager.num_ci]
    next_c_y_vars = next_c_vars[manager.num_ci : 2 * manager.num_ci]
    next_c_theta_vars = next_c_vars[2 * manager.num_ci : 3 * manager.num_ci]
    prev_t_var = func_vars[-1]

    prev_sigma = manager.compute_sigma_i_exp(
        c_x_i_vars=prev_c_x_vars,
        c_y_i_vars=prev_c_y_vars,
        c_theta_i_vars=prev_c_theta_vars,
        t_exp=prev_t_var,
        derivative=derivative,
    )
    next_sigma = manager.compute_sigma_i_exp(
        c_x_i_vars=next_c_x_vars,
        c_y_i_vars=next_c_y_vars,
        c_theta_i_vars=next_c_theta_vars,
        t_exp=0.0,
        derivative=derivative,
    )
    # This can be sneakily bug prone. If the derivative is 0, we want to
    # find the normalized angle diff while computing the pose delta
    # IF derivative > 0, we shouldn't do any angle wrapping
    # and so we just take the regular difference.
    if derivative == 0:
        return _compute_pose_diff(
            pose_vector=prev_sigma,
            expected_pose_vector=next_sigma,
        )
    else:
        return prev_sigma - next_sigma


def kinematic_constraint_func(
    func_vars: np.ndarray,
    manager: RunitoVariableManager,
) -> np.ndarray:
    assert func_vars.shape == (3 * manager.num_c + manager.params.M,)

    constraint_vector = []

    for i in range(manager.params.M):
        c_x_i_vars = manager.get_c_x_i_vars(all_vars=func_vars, i=i)
        c_y_i_vars = manager.get_c_y_i_vars(all_vars=func_vars, i=i)
        c_theta_i_vars = manager.get_c_theta_i_vars(all_vars=func_vars, i=i)
        t_i_var = manager.get_t_i_var(func_vars, i)

        for j in range(manager.params.n):
            t_ijl = manager.compute_t_ijl_exp(
                t_i_var=t_i_var,
                j=j,
                l=0,
            )

            sigma_i = manager.compute_sigma_i_exp(
                c_x_i_vars=c_x_i_vars,
                c_y_i_vars=c_y_i_vars,
                c_theta_i_vars=c_theta_i_vars,
                t_exp=t_ijl,
                derivative=0,
            )
            sigma_i_dot = manager.compute_sigma_i_exp(
                c_x_i_vars=c_x_i_vars,
                c_y_i_vars=c_y_i_vars,
                c_theta_i_vars=c_theta_i_vars,
                t_exp=t_ijl,
                derivative=1,
            )

            # TODO: We get a divide by zero error here if both xdot and ydot end up being 0.
            # So we add an epsilon.
            epsilon = 0
            if sigma_i_dot[0] == 0 and sigma_i_dot[1] == 0:
                epsilon = 1e-12
            v = np.sqrt(sigma_i_dot[0] ** 2 + sigma_i_dot[1] ** 2 + epsilon)
            constraint_vector.append(sigma_i_dot[0] - v * np.cos(sigma_i[2]))
            constraint_vector.append(sigma_i_dot[1] - v * np.sin(sigma_i[2]))

    return np.array(constraint_vector)


def obstacle_constraint_func(
    func_vars: np.ndarray,
    footprint: PolygonXYArray,
    emap2d: EnvironmentMap2D,
    # Signed distance can be computed from the emap, but we have a separate argument
    # in order to avoid recomputing it each time the constraint is evaluated.
    signed_distance_map: MatrixMNf32,
    obstacle_clearance: float,
    manager: RunitoVariableManager,
) -> np.ndarray:
    assert func_vars.shape == (3 * manager.num_c + manager.params.M,)

    constraint_vector = []
    h, w = signed_distance_map.shape

    for i in range(manager.params.M):
        c_x_i_vars = manager.get_c_x_i_vars(all_vars=func_vars, i=i)
        c_y_i_vars = manager.get_c_y_i_vars(all_vars=func_vars, i=i)
        c_theta_i_vars = manager.get_c_theta_i_vars(all_vars=func_vars, i=i)
        t_i_var = manager.get_t_i_var(func_vars, i)

        for j in range(manager.params.n):
            sigma_i = manager.compute_sigma_i_exp(
                c_x_i_vars=c_x_i_vars,
                c_y_i_vars=c_y_i_vars,
                c_theta_i_vars=c_theta_i_vars,
                t_exp=manager.compute_t_ijl_exp(
                    t_i_var=t_i_var,
                    j=j,
                    l=0,
                ),
            )
            x_ij = sigma_i[0]
            y_ij = sigma_i[1]
            theta_ij = sigma_i[2]

            # Need to spell this transform out so that we get gradient information.
            # TODO: Support Autodiff in transform utils.
            rotation_matrix = np.array(
                [
                    [np.cos(theta_ij), -np.sin(theta_ij)],
                    [np.sin(theta_ij), np.cos(theta_ij)],
                ]
            )
            transformed_footprint = footprint @ rotation_matrix.T + np.array([x_ij, y_ij])

            for footprint_i in range(transformed_footprint.shape[0]):
                # Bilinear interpolation.
                fx = transformed_footprint[footprint_i, 0]
                fy = transformed_footprint[footprint_i, 1]

                px_x, px_y = emap2d.xy_to_px(
                    xy=(fx, fy),
                    output_as_float=True,
                )

                px_x0, px_y0 = int(np.floor(px_x)), int(np.floor(px_y))
                # TODO: Add clipping to emap2d
                px_x0 = np.clip(px_x0, 0, w - 1)
                px_y0 = np.clip(px_y0, 0, h - 1)

                px_x1, px_y1 = px_x0 + 1, px_y0 + 1
                px_x1 = np.clip(px_x1, 0, w - 1)
                px_y1 = np.clip(px_y1, 0, h - 1)

                # Note: px_x, py_x are the x and y pixel coordinates (bottom left origin).
                # To access the array, we need to flip the y coordinate (due to top left origin)
                # and also access it as h - px_y, px_x
                sd1 = signed_distance_map[h - 1 - px_y0, px_x0]
                sd2 = signed_distance_map[h - 1 - px_y0, px_x1]
                sd3 = signed_distance_map[h - 1 - px_y1, px_x0]
                sd4 = signed_distance_map[h - 1 - px_y1, px_x1]

                dx = px_x - px_x0
                dy = px_y - px_y0

                sd5 = sd1 * (1 - dx) + sd2 * dx
                sd6 = sd3 * (1 - dx) + sd4 * dx
                sd = sd5 * (1 - dy) + sd6 * dy

                constraint_vector.append(sd - obstacle_clearance)

    return np.array(constraint_vector)
