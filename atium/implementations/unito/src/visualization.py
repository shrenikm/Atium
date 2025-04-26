import matplotlib.pyplot as plt
import numpy as np

from atium.core.utils.custom_types import DecisionVariablesVector
from atium.implementations.unito.src.unito_utils import UnitoInputs
from atium.implementations.unito.src.unito_variable_manager import UnitoVariableManager


def visualize_unito_result(
    manager: UnitoVariableManager,
    unito_inputs: UnitoInputs,
    all_vars_solution: DecisionVariablesVector,
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

    x_values = []
    y_values = []
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

        x_i = manager.compute_x_i_exp(
            all_vars=all_vars_solution,
            initial_x=unito_inputs.initial_state_inputs.initial_xy[0],
            i=i,
        )
        y_i = manager.compute_y_i_exp(
            all_vars=all_vars_solution,
            initial_y=unito_inputs.initial_state_inputs.initial_xy[1],
            i=i,
        )
        x_values.append(x_i)
        y_values.append(y_i)

        ax1.plot(t_values, theta_values, color=ms_color, label=f"Segment {i}: theta")
        ax2.plot(t_values, s_values, color=ms_color, label=f"Segment {i}: s")

    xf = manager.compute_x_i_exp(
        all_vars=all_vars_solution,
        initial_x=unito_inputs.initial_state_inputs.initial_xy[0],
        i=manager.params.M,
    )
    yf = manager.compute_y_i_exp(
        all_vars=all_vars_solution,
        initial_y=unito_inputs.initial_state_inputs.initial_xy[1],
        i=manager.params.M,
    )
    x_values.append(xf)
    y_values.append(yf)

    ax3.plot(x_values, y_values, color=xy_color, label="xy")
    for k in range(len(x_values)):
        ax3.plot(x_values[k], y_values[k], "o", color=xy_color)

    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.tight_layout()
    plt.show()
