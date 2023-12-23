import attr
from typing import Dict, List, Sequence, Tuple
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np

from algorithms.multi_vehicle_mip.implementation.definitions import (
    MVMIPObstacle,
    MVMIPOptimizationParams,
    MVMIPResult,
    MVMIPVehicle,
)


@attr.frozen
class _MVMIPAnimationColors:
    vehicle_colors: List[str]
    vehicle_start_color: str
    vehicle_end_color: str
    vehicle_control_color: str
    vehicle_clearance_color: str
    vehicle_trajectory_color: str

    obstacle_colors: List[str]
    obstacle_clearance_color: str


@attr.frozen
class _MVMIPAnimationElements:
    vehicle_core_map: Dict[int, plt.Line2D]
    vehicle_clearance_map: Dict[int, patches.Rectangle]
    vehicle_control_map: Dict[int, patches.FancyArrow]
    vehicle_trajectory_map: Dict[int, plt.Line2D]

    obstacle_core_map: Dict[int, patches.Polygon]
    obstacle_clearance_map: Dict[int, patches.Polygon]


def _setup_figure(
    vehicles: Sequence[MVMIPVehicle],
    padding_m: float = 1.0,
) -> Tuple[plt.Figure, plt.Axes]:

    fig = plt.figure("MVMIP")
    ax = fig.gca()

    # Set axis labels and limits
    ax.set_xlabel("x (m)")
    ax.set_ylabel("x (m)")

    # Compute world limits from all vehicles.
    all_vehicle_state_min = np.min(
        np.vstack([vehicle.optimization_params.state_min for vehicle in vehicles]),
        axis=0,
    )
    all_vehicle_state_max = np.max(
        np.vstack([vehicle.optimization_params.state_max for vehicle in vehicles]),
        axis=0,
    )
    all_vehicle_state_min -= padding_m
    all_vehicle_state_max += padding_m
    ax.set_xlim(all_vehicle_state_min[0], all_vehicle_state_max[0])
    ax.set_ylim(all_vehicle_state_min[1], all_vehicle_state_max[1])
    ax.set_aspect(1.0)

    return fig, ax


def _initialize_animation_colors(
    vehicles: Sequence[MVMIPVehicle],
    obstacles: Sequence[MVMIPObstacle],
) -> _MVMIPAnimationColors:

    available_vehicle_colors = ["green", "cyan", "orange"]
    available_obstacle_colors = ["lightcoral"]
    vehicle_colors = [
        available_vehicle_colors[i % len(available_vehicle_colors)]
        for i in range(len(vehicles))
    ]
    obstacle_colors = [
        available_obstacle_colors[i % len(available_obstacle_colors)]
        for i in range(len(obstacles))
    ]

    vehicle_start_color = "skyblue"
    vehicle_end_color = "mediumseagreen"
    vehicle_control_color = "mediumpurple"
    vehicle_clearance_color = "cadetblue"
    vehicle_trajectory_color = "lightgray"

    obstacle_clearance_color = "rosybrown"

    return _MVMIPAnimationColors(
        vehicle_colors=vehicle_colors,
        vehicle_start_color=vehicle_start_color,
        vehicle_end_color=vehicle_end_color,
        vehicle_control_color=vehicle_control_color,
        vehicle_clearance_color=vehicle_clearance_color,
        vehicle_trajectory_color=vehicle_trajectory_color,
        obstacle_colors=obstacle_colors,
        obstacle_clearance_color=obstacle_clearance_color,
    )


def _draw_fixed_elements(
    ax: plt.Axes,
    animation_colors: _MVMIPAnimationColors,
    vehicles: Sequence[MVMIPVehicle],
) -> None:

    for vehicle_id, vehicle in enumerate(vehicles):
        sx, sy = vehicle.dynamics.initial_state[:2]
        fx, fy = vehicle.dynamics.final_state[:2]

        ax.plot(sx, sy, marker="o", color=animation_colors.vehicle_start_color)
        ax.plot(fx, fy, marker="o", color=animation_colors.vehicle_end_color)


def _create_animation_elements(
    ax: plt.Axes,
    animation_colors: _MVMIPAnimationColors,
    mvmip_params: MVMIPOptimizationParams,
    vehicles: Sequence[MVMIPVehicle],
    obstacles: Sequence[MVMIPObstacle],
) -> _MVMIPAnimationElements:

    vehicle_core_map = {}
    vehicle_clearance_map = {}
    vehicle_control_map = {}
    vehicle_trajectory_map = {}

    obstacle_core_map = {}
    obstacle_clearance_map = {}

    nt, dt = mvmip_params.num_time_steps, mvmip_params.dt

    # Plot the initial positions of the vehicles and obstacles.
    # Also sets up the patch elements for later animation.
    for obstacle_id, obstacle in enumerate(obstacles):
        corner_points_xy = obstacle.ordered_corner_points_xy(
            time_step_id=0,
            num_time_steps=nt,
            dt=dt,
            add_clearance=False,
        )
        clearance_corner_points_xy = obstacle.ordered_corner_points_xy(
            time_step_id=0,
            num_time_steps=nt,
            dt=dt,
            add_clearance=True,
        )
        core_polygon = patches.Polygon(
            xy=corner_points_xy,
            color=animation_colors.obstacle_colors[obstacle_id],
            fill=True,
        )
        clearance_polygon = patches.Polygon(
            xy=clearance_corner_points_xy,
            color=animation_colors.obstacle_clearance_color,
            fill=False,
            alpha=0.5,
        )
        ax.add_patch(core_polygon)
        ax.add_patch(clearance_polygon)

        obstacle_core_map[obstacle_id] = core_polygon
        obstacle_clearance_map[obstacle_id] = clearance_polygon

    for vehicle_id, vehicle in enumerate(vehicles):
        x, y = vehicle.dynamics.initial_state[:2]
        c_m = vehicle.dynamics.clearance_m

        core_line = plt.Line2D(
            xdata=[x],
            ydata=[y],
            marker="o",
            markersize=7,
            color=animation_colors.vehicle_colors[vehicle_id],
        )
        clearance_rect = patches.Rectangle(
            xy=(x - c_m, y - c_m),
            width=2 * c_m,
            height=2 * c_m,
            color=animation_colors.vehicle_clearance_color,
            fill=False,
            alpha=0.5,
        )
        control_arrow = patches.FancyArrow(
            x=x,
            y=y,
            dx=0,  # 0 control initially
            dy=0,  # 0 control initially
            length_includes_head=True,
            color=animation_colors.vehicle_control_color,
        )
        trajectory_line = plt.Line2D(
            xdata=[x],
            ydata=[y],
            linestyle="dotted",
            color=animation_colors.vehicle_trajectory_color,
        )
        ax.add_patch(trajectory_line)
        ax.add_patch(core_line)
        ax.add_patch(clearance_rect)
        ax.add_patch(control_arrow)

        vehicle_core_map[vehicle_id] = core_line
        vehicle_clearance_map[vehicle_id] = clearance_rect
        vehicle_control_map[vehicle_id] = control_arrow
        vehicle_trajectory_map[vehicle_id] = trajectory_line

        return _MVMIPAnimationElements(
            vehicle_core_map=vehicle_core_map,
            vehicle_clearance_map=vehicle_clearance_map,
            vehicle_control_map=vehicle_control_map,
            vehicle_trajectory_map=vehicle_trajectory_map,
            obstacle_core_map=obstacle_core_map,
            obstacle_clearance_map=obstacle_clearance_map,
        )


def visualize_mvmip_result(
    mvmip_result: MVMIPResult,
) -> None:

    nt = mvmip_result.mvmip_params.num_time_steps
    dt = mvmip_result.mvmip_params.dt
    vst_map = mvmip_result.vehicle_state_trajectory_map
    vct_map = mvmip_result.vehicle_control_trajectory_map
    vehicles = mvmip_result.vehicles
    obstacles = mvmip_result.obstacles
    num_vehicles, num_obstacles = len(vehicles), len(obstacles)

    fig, ax = _setup_figure(vehicles=vehicles)

    animation_colors = _initialize_animation_colors(
        vehicles=vehicles,
        obstacles=obstacles,
    )

    _draw_fixed_elements(
        ax=ax,
        animation_colors=animation_colors,
        vehicles=vehicles,
    )

    animation_elements = _create_animation_elements(
        ax=ax,
        animation_colors=animation_colors,
        mvmip_params=mvmip_result.mvmip_params,
        vehicles=vehicles,
        obstacles=obstacles,
    )

    def anim_update(time_step_id):
        ax.set_title(f"MVMIP time step: {time_step_id}")
        for obstacle_id, obstacle in enumerate(obstacles):
            corner_points_xy = obstacle.ordered_corner_points_xy(
                time_step_id=time_step_id,
                num_time_steps=nt,
                dt=dt,
                add_clearance=False,
            )
            clearance_corner_points_xy = obstacle.ordered_corner_points_xy(
                time_step_id=time_step_id,
                num_time_steps=nt,
                dt=dt,
                add_clearance=True,
            )

            animation_elements.obstacle_core_map[obstacle_id].set_xy(corner_points_xy)
            animation_elements.obstacle_clearance_map[obstacle_id].set_xy(
                clearance_corner_points_xy
            )

        for vehicle_id, vehicle in enumerate(vehicles):
            state_trajectory = vst_map[vehicle_id]
            control_trajectory = vct_map[vehicle_id]

            x, y = state_trajectory[time_step_id][:2]
            c_m = vehicle.dynamics.clearance_m

            animation_elements.vehicle_core_map[vehicle_id].set_xdata(x)
            animation_elements.vehicle_core_map[vehicle_id].set_ydata(y)
            animation_elements.vehicle_clearance_map[vehicle_id].set_xy(
                (x - c_m, y - c_m)
            )
            animation_elements.vehicle_control_map[vehicle_id].set_data(
                x=x,
                y=y,
                # time_step_id is in [0, nt], but control isn't defined for nt
                dx=control_trajectory[min(time_step_id, nt - 1)][0] * dt,
                dy=control_trajectory[min(time_step_id, nt - 1)][1] * dt,
            )
            animation_elements.vehicle_trajectory_map[vehicle_id].set_xdata(
                state_trajectory[: time_step_id + 1, 0]
            )
            animation_elements.vehicle_trajectory_map[vehicle_id].set_ydata(
                state_trajectory[: time_step_id + 1, 1]
            )

    _ = anim.FuncAnimation(
        fig=fig,
        func=anim_update,
        frames=nt + 1,
        interval=100,
        repeat=True,
    )

    plt.show()
