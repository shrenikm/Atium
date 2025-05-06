import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from atium.core.utils.color_utils import ColorType
from atium.core.utils.custom_types import DecisionVariablesVector
from atium.core.utils.transformation_utils import transform_points_2d
from atium.experiments.runito.src.runito_utils import RunitoInputs
from atium.experiments.runito.src.runito_variable_manager import RunitoVariableManager


def visualize_runito_result(
    manager: RunitoVariableManager,
    unito_inputs: RunitoInputs,
    all_vars_solution: DecisionVariablesVector,
    draw_heading: bool = False,
) -> None:
    fix, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

    # Prep the axes for plotting.
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel("x (rad)")
    ax1.set_title("X")

    ax2.set_xlabel("t (s)")
    ax2.set_ylabel("y (rad)")
    ax2.set_title("Y")

    ax3.set_xlabel("t (s)")
    ax3.set_ylabel("theta (rad)")
    ax3.set_title("Theta")

    ax4.set_xlabel("x (m)")
    ax4.set_ylabel("y (m)")
    ax4.set_title("XY")
    ax4.set_aspect("equal")

    # For the individual sigma curves, each successive curve is plotted using a different color.

    sigma_colors = ["lightcoral", "mediumseagreen", "turquoise"]
    xy_color = "slateblue"

    # Initial pose
    initial_pose_vector = unito_inputs.initial_state_inputs.initial_pose.to_vector()
    final_pose_vector = unito_inputs.final_state_inputs.final_pose.to_vector()

    # All x and y values.
    full_x_values = [initial_pose_vector[0]]
    full_y_values = [initial_pose_vector[1]]
    full_theta_values = [initial_pose_vector[2]]

    # X and y values at the end of each segment.
    x_segment_values = []
    y_segment_values = []
    t_values = []

    for i in range(manager.params.M):
        sigma_color = sigma_colors[i % len(sigma_colors)]

        c_x_i_values = manager.get_c_x_i_vars(all_vars_solution, i)
        c_y_i_values = manager.get_c_y_i_vars(all_vars_solution, i)
        c_theta_i_values = manager.get_c_theta_i_vars(all_vars_solution, i)
        t_i_value = manager.get_t_i_var(all_vars_solution, i)

        # Each segment resets t to 0 and goes through T_i
        # If we don't add the value up until T_{i-1} to the current t_sample,
        # the plot will be discontinuous in x.
        t_offset = np.sum(manager.get_t_vars(all_vars_solution)[:i])

        x_values, y_values, theta_values, t_values = [], [], [], []
        for t_sample in np.linspace(0, t_i_value, 10):
            sigma_i = manager.compute_sigma_i_exp(
                c_x_i_vars=c_x_i_values,
                c_y_i_vars=c_y_i_values,
                c_theta_i_vars=c_theta_i_values,
                t_exp=t_sample,
                derivative=0,
            )
            x_values.append(sigma_i[0])
            y_values.append(sigma_i[1])
            theta_values.append(sigma_i[2])
            t_values.append(t_sample + t_offset)

            full_x_values.append(sigma_i[0])
            full_y_values.append(sigma_i[1])
            full_theta_values.append(sigma_i[2])

        x_segment_values.append(x_values[-1])
        y_segment_values.append(y_values[-1])

        ax1.plot(t_values, x_values, color=sigma_color, label=f"Segment {i}: s")
        ax2.plot(t_values, y_values, color=sigma_color, label=f"Segment {i}: s")
        ax3.plot(t_values, theta_values, color=sigma_color, label=f"Segment {i}: theta")

        ax1.plot(t_values[-1], x_values[-1], "o", color=sigma_color)
        ax2.plot(t_values[-1], y_values[-1], "o", color=sigma_color)
        ax3.plot(t_values[-1], theta_values[-1], "o", color=sigma_color)

    # Draw the environment map.
    emap_size_xy = unito_inputs.emap2d.size_xy
    ax4.imshow(
        unito_inputs.emap2d.create_rgb_viz(color_type=ColorType.RGB),
        extent=[0, emap_size_xy[0], 0, emap_size_xy[1]],
        origin="lower",
    )

    # Draw the footprint at the initial position.
    transformed_footprint = transform_points_2d(
        points=unito_inputs.footprint,
        translation=initial_pose_vector[:2],
        rotation=initial_pose_vector[2],
    )
    polygon = Polygon(
        transformed_footprint,
        closed=True,
        edgecolor="brown",
        fill=False,
        linewidth=1,
    )
    ax4.add_patch(polygon)

    # Draw the footprint at each sampling point.
    for i in range(len(full_x_values)):
        transformed_footprint = transform_points_2d(
            points=unito_inputs.footprint,
            translation=[full_x_values[i], full_y_values[i]],
            rotation=full_theta_values[i],
        )
        polygon = Polygon(
            transformed_footprint,
            closed=True,
            edgecolor="gray",
            alpha=0.5,
            fill=False,
            linewidth=1,
        )
        ax4.add_patch(polygon)

    # Plot xy
    ax4.plot(full_x_values, full_y_values, color=xy_color, label="xy")
    for k in range(len(x_segment_values)):
        ax4.plot(x_segment_values[k], y_segment_values[k], "o", color=xy_color)

    # Plot heading.
    if draw_heading:
        ax4.quiver(
            full_x_values,
            full_y_values,
            np.cos(full_theta_values),
            np.sin(full_theta_values),
            angles="xy",
            scale_units="xy",
            scale=1,
            color=xy_color,
        )

    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.tight_layout()
    plt.show()
