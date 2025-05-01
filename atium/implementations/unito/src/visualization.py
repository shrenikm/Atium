import matplotlib.pyplot as plt
import numpy as np

from atium.core.utils.color_utils import ColorType
from atium.core.utils.custom_types import DecisionVariablesVector
from atium.implementations.unito.src.unito_utils import UnitoInputs
from atium.implementations.unito.src.unito_variable_manager import UnitoVariableManager


def visualize_unito_result(
    manager: UnitoVariableManager,
    unito_inputs: UnitoInputs,
    all_vars_solution: DecisionVariablesVector,
    draw_heading: bool = False,
) -> None:
    fix, (ax1, ax2, ax3) = plt.subplots(1, 3)

    # Prep the axes for plotting.
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel("theta (rad)")
    ax1.set_title("Theta")
    ax2.set_xlabel("t (s)")
    ax2.set_ylabel("s (m)")
    ax2.set_title("S")
    ax3.set_xlabel("x (m)")
    ax3.set_ylabel("y (m)")
    ax3.set_title("XY")
    ax3.set_aspect("equal")

    # For the theta and s curves, each successive curve is plotted using a different color.

    ms_colors = ["lightcoral", "mediumseagreen", "turquoise"]
    xy_color = "slateblue"

    # All x and y values.
    x_values = [unito_inputs.initial_state_inputs.initial_xy[0]]
    y_values = [unito_inputs.initial_state_inputs.initial_xy[1]]
    heading_values = [
        unito_inputs.initial_state_inputs.initial_ms_map[0][0]
        if unito_inputs.initial_state_inputs.initial_ms_map
        else 0.0
    ]
    # X and y values at the end of each segment.
    x_segment_values = []
    y_segment_values = []
    t_values = []

    for i in range(manager.params.M):
        ms_color = ms_colors[i % len(ms_colors)]
        c_theta_i_values = manager.get_c_theta_i_vars(all_vars_solution, i)
        c_s_i_values = manager.get_c_s_i_vars(all_vars_solution, i)
        t_i_value = manager.get_t_i_var(all_vars_solution, i)
        # Each segment resets t to 0 and goes through T_i
        # If we don't add the value up until T_{i-1} to the current t_sample,
        # the plot will be discontinuous in x.
        t_offset = np.sum(manager.get_t_vars(all_vars_solution)[:i])

        theta_values, s_values, t_values = [], [], []
        for t_sample in np.linspace(0, t_i_value, 10):
            sigma_i = manager.compute_sigma_i_exp(
                c_theta_i_vars=c_theta_i_values,
                c_s_i_vars=c_s_i_values,
                t_exp=t_sample,
                derivative=0,
            )
            theta_values.append(sigma_i[0])
            s_values.append(sigma_i[1])
            t_values.append(t_sample + t_offset)

        for j in range(manager.params.n):
            x_ij = manager.compute_x_ij_exp(
                all_vars=all_vars_solution,
                initial_x=unito_inputs.initial_state_inputs.initial_xy[0],
                i=i,
                j=j,
            )
            y_ij = manager.compute_y_ij_exp(
                all_vars=all_vars_solution,
                initial_y=unito_inputs.initial_state_inputs.initial_xy[1],
                i=i,
                j=j,
            )
            x_values.append(x_ij)
            y_values.append(y_ij)
            heading_values.append(
                manager.compute_sigma_i_exp(
                    c_theta_i_vars=c_theta_i_values,
                    c_s_i_vars=c_s_i_values,
                    t_exp=manager.compute_t_ijl_exp(
                        t_i_var=t_i_value,
                        j=j,
                        l=0,
                    ),
                )[0]
            )
            if j == manager.params.n - 1:
                x_segment_values.append(x_ij)
                y_segment_values.append(y_ij)

        ax1.plot(t_values, theta_values, color=ms_color, label=f"Segment {i}: theta")
        ax2.plot(t_values, s_values, color=ms_color, label=f"Segment {i}: s")
        ax1.plot(t_values[-1], theta_values[-1], "o", color=ms_color)
        ax2.plot(t_values[-1], s_values[-1], "o", color=ms_color)

    # Draw the environment map.
    emap_size_xy = unito_inputs.emap2d.size_xy
    ax3.imshow(
        unito_inputs.emap2d.create_rgb_viz(color_type=ColorType.RGB),
        extent=[0, emap_size_xy[0], 0, emap_size_xy[1]],
        origin="lower",
    )

    # Plot xy
    ax3.plot(x_values, y_values, color=xy_color, label="xy")
    for k in range(len(x_segment_values)):
        ax3.plot(x_segment_values[k], y_segment_values[k], "o", color=xy_color)

    # Plot heading.
    if draw_heading:
        ax3.quiver(
            x_values,
            y_values,
            np.cos(heading_values),
            np.sin(heading_values),
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
