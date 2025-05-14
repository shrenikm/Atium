from typing import Self

import attr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon

from atium.core.definitions.concrete_states import Pose2D
from atium.core.utils.color_utils import ColorType
from atium.core.utils.custom_types import DecisionVariablesVector
from atium.core.utils.transformation_utils import transform_points_2d
from atium.experiments.runito.src.runito_utils import RunitoInputs
from atium.experiments.runito.src.runito_variable_manager import RunitoVariableManager


@attr.frozen
class RunitoVisualizationData:
    x_i_values: list[list[float]]
    y_i_values: list[list[float]]
    theta_i_values: list[list[float]]
    t_i_values: list[list[float]]
    x_values: list[float]
    y_values: list[float]
    theta_values: list[float]

    @classmethod
    def create(
        cls,
        manager: RunitoVariableManager,
        all_vars: DecisionVariablesVector,
        num_samples_per_segment: int = 10,
    ) -> Self:
        x_i_values, y_i_values, theta_i_values, t_i_values = [], [], [], []
        x_values, y_values, theta_values = [], [], []

        for i in range(manager.params.M):
            c_x_i_values = manager.get_c_x_i_vars(all_vars, i)
            c_y_i_values = manager.get_c_y_i_vars(all_vars, i)
            c_theta_i_values = manager.get_c_theta_i_vars(all_vars, i)
            t_i_value = manager.get_t_i_var(all_vars, i)

            # Each segment resets t to 0 and goes through T_i
            # If we don't add the value up until T_{i-1} to the current t_sample,
            # the plot will be discontinuous in x.
            t_offset = np.sum(manager.get_t_vars(all_vars)[:i])

            xv, yv, thetav, tv = [], [], [], []
            for t_sample in np.linspace(0, t_i_value, num_samples_per_segment):
                sigma_i = manager.compute_sigma_i_exp(
                    c_x_i_vars=c_x_i_values,
                    c_y_i_vars=c_y_i_values,
                    c_theta_i_vars=c_theta_i_values,
                    t_exp=t_sample,
                    derivative=0,
                )
                xv.append(sigma_i[0])
                yv.append(sigma_i[1])
                thetav.append(sigma_i[2])
                tv.append(t_sample + t_offset)

            x_i_values.append(xv)
            y_i_values.append(yv)
            theta_i_values.append(thetav)
            t_i_values.append(tv)

            for j in range(manager.params.n):
                t_ij_vals = [
                    manager.compute_t_ijl_exp(
                        t_i_var=t_i_value,
                        j=j,
                        l=l,
                    )
                    for l in [0, 1, 2]
                ]
                sigma_i_vals = [
                    manager.compute_sigma_i_exp(
                        c_x_i_vars=c_x_i_values,
                        c_y_i_vars=c_y_i_values,
                        c_theta_i_vars=c_theta_i_values,
                        t_exp=t_ijl,
                    )
                    for t_ijl in t_ij_vals
                ]
                for sigma_i in sigma_i_vals:
                    x_values.append(sigma_i[0])
                    y_values.append(sigma_i[1])
                    theta_values.append(sigma_i[2])

        return cls(
            x_i_values=x_i_values,
            y_i_values=y_i_values,
            theta_i_values=theta_i_values,
            t_i_values=t_i_values,
            x_values=x_values,
            y_values=y_values,
            theta_values=theta_values,
        )


@attr.define
class RunitoVisualizationAxes:
    # For the individual sigma curves, each successive curve is plotted using a different color.
    SIGMA_COLORS = ["lightcoral", "mediumseagreen", "turquoise"]
    XY_COLOR = "slateblue"
    INITIAL_FOOTPRINT_COLOR = "brown"
    TRAJECTORY_FOOTPRINT_COLOR = "gray"

    fig: Figure
    x_axes: Axes
    y_axes: Axes
    theta_axes: Axes
    guess_xy_axes: Axes
    solution_xy_axes: Axes

    @classmethod
    def create(cls) -> Self:
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1])

        # Top row: x, y, theta
        x_axes = fig.add_subplot(gs[0, 0])
        y_axes = fig.add_subplot(gs[0, 1])
        theta_axes = fig.add_subplot(gs[0, 2])

        # Bottom row: guess and solution xy
        guess_xy_axes = fig.add_subplot(gs[1, 0])
        solution_xy_axes = fig.add_subplot(gs[1, 2])

        x_axes.set_xlabel("t (s)")
        x_axes.set_ylabel("x (rad)")
        x_axes.set_title("X")

        y_axes.set_xlabel("t (s)")
        y_axes.set_ylabel("y (rad)")
        y_axes.set_title("Y")

        theta_axes.set_xlabel("t (s)")
        theta_axes.set_ylabel("theta (rad)")
        theta_axes.set_title("Theta")

        guess_xy_axes.set_xlabel("x (m)")
        guess_xy_axes.set_ylabel("y (m)")
        guess_xy_axes.set_title("Guess XY")

        solution_xy_axes.set_xlabel("x (m)")
        solution_xy_axes.set_ylabel("y (m)")
        solution_xy_axes.set_title("Solution XY")

        return cls(
            fig=fig,
            x_axes=x_axes,
            y_axes=y_axes,
            theta_axes=theta_axes,
            guess_xy_axes=guess_xy_axes,
            solution_xy_axes=solution_xy_axes,
        )

    def plot_sigma(
        self,
        manager: RunitoVariableManager,
        data: RunitoVisualizationData,
    ) -> None:
        # Validate the data.
        assert len(data.x_i_values) == manager.params.M
        assert len(data.y_i_values) == manager.params.M
        assert len(data.theta_i_values) == manager.params.M
        assert len(data.t_i_values) == manager.params.M
        assert all(len(xv) == len(data.x_i_values[0]) for xv in data.x_i_values)
        assert all(len(yv) == len(data.y_i_values[0]) for yv in data.y_i_values)
        assert all(len(thetav) == len(data.theta_i_values[0]) for thetav in data.theta_i_values)
        assert all(len(tv) == len(data.t_i_values[0]) for tv in data.t_i_values)

        for i in range(manager.params.M):
            sigma_color = self.SIGMA_COLORS[i % len(self.SIGMA_COLORS)]

            self.x_axes.plot(data.t_i_values[i], data.x_i_values[i], color=sigma_color, label=f"Segment {i}: s")
            self.x_axes.plot(data.t_i_values[i][-1], data.x_i_values[i][-1], "o", color=sigma_color)

            self.y_axes.plot(data.t_i_values[i], data.y_i_values[i], color=sigma_color, label=f"Segment {i}: s")
            self.y_axes.plot(data.t_i_values[i][-1], data.y_i_values[i][-1], "o", color=sigma_color)

            self.theta_axes.plot(
                data.t_i_values[i], data.theta_i_values[i], color=sigma_color, label=f"Segment {i}: theta"
            )
            self.theta_axes.plot(data.t_i_values[i][-1], data.theta_i_values[i][-1], "o", color=sigma_color)

    def _plot_single_xy(
        self,
        ax: Axes,
        manager: RunitoVariableManager,
        unito_inputs: RunitoInputs,
        data: RunitoVisualizationData,
        draw_heading: bool = False,
    ) -> None:
        initial_pose_vector = unito_inputs.initial_state_inputs.initial_pose.to_vector()

        # Draw the environment map.
        emap_size_xy = unito_inputs.emap2d.size_xy
        ax.imshow(
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
            edgecolor=self.INITIAL_FOOTPRINT_COLOR,
            fill=False,
            linewidth=1,
        )
        ax.add_patch(polygon)

        # Draw the footprint at each sampling point.
        for i in range(len(data.x_values)):
            transformed_footprint = transform_points_2d(
                points=unito_inputs.footprint,
                translation=[data.x_values[i], data.y_values[i]],
                rotation=data.theta_values[i],
            )
            polygon = Polygon(
                transformed_footprint,
                closed=True,
                edgecolor=self.TRAJECTORY_FOOTPRINT_COLOR,
                alpha=0.5,
                fill=False,
                linewidth=1,
            )
            ax.add_patch(polygon)

        # Plot the xy trajectory.
        ax.plot(data.x_values, data.y_values, color=self.XY_COLOR, label="xy")
        for i in range(manager.params.M):
            ax.plot(data.x_i_values[i][-1], data.y_i_values[i][-1], "o", color=self.XY_COLOR)

        # Plot heading.
        if draw_heading:
            ax.quiver(
                data.x_values,
                data.y_values,
                np.cos(data.theta_values),
                np.sin(data.theta_values),
                angles="xy",
                scale_units="xy",
                scale=5,
                color=self.XY_COLOR,
            )

    def plot_xy(
        self,
        manager: RunitoVariableManager,
        unito_inputs: RunitoInputs,
        guess_data: RunitoVisualizationData,
        solution_data: RunitoVisualizationData,
        draw_heading: bool = False,
    ) -> None:
        self._plot_single_xy(
            ax=self.guess_xy_axes,
            manager=manager,
            unito_inputs=unito_inputs,
            data=guess_data,
            draw_heading=draw_heading,
        )
        self._plot_single_xy(
            ax=self.solution_xy_axes,
            manager=manager,
            unito_inputs=unito_inputs,
            data=solution_data,
            draw_heading=draw_heading,
        )

    def legend(self) -> None:
        self.x_axes.legend()
        self.y_axes.legend()
        self.theta_axes.legend()
        self.guess_xy_axes.legend()
        self.solution_xy_axes.legend()


def visualize_runito_result(
    manager: RunitoVariableManager,
    unito_inputs: RunitoInputs,
    all_vars_guess: DecisionVariablesVector,
    all_vars_solution: DecisionVariablesVector,
    draw_heading: bool = False,
) -> None:
    axes = RunitoVisualizationAxes.create()

    guess_data = RunitoVisualizationData.create(
        manager=manager,
        all_vars=all_vars_guess,
    )
    solution_data = RunitoVisualizationData.create(
        manager=manager,
        all_vars=all_vars_solution,
    )

    # Draw the sigma curves with the solution values.
    axes.plot_sigma(manager=manager, data=solution_data)

    # Draw the xy trajectories with both the guess and solution values.
    axes.plot_xy(
        manager=manager,
        unito_inputs=unito_inputs,
        guess_data=guess_data,
        solution_data=solution_data,
        draw_heading=draw_heading,
    )
    axes.legend()

    plt.tight_layout()
    plt.show()
