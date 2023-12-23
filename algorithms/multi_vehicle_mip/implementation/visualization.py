import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np

from algorithms.multi_vehicle_mip.implementation.definitions import MVMIPResult


def visualize_mvmip_result(
    mvmip_result: MVMIPResult,
) -> None:

    nt = mvmip_result.mvmip_params.num_time_steps
    vst_map = mvmip_result.vehicle_state_trajectory_map
    vct_map = mvmip_result.vehicle_control_trajectory_map
    vehicles = mvmip_result.vehicles
    obstacles = mvmip_result.obstacles
    num_vehicles, num_obstacles = len(vehicles), len(obstacles)

    fig = plt.figure()
    ax = fig.gca()

    # Set axis labels and limits
    ax.set_xlabel("x (m)")
    ax.set_ylabel("x (m)")
    # Compute world limits from all vehicles.
    padding_m = 1.0
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

    # Display variables
    available_vehicle_colors = ["green", "cyan", "orange"]
    vehicle_colors = [
        available_vehicle_colors[i % len(available_vehicle_colors)]
        for i in range(num_vehicles)
    ]

    vehicle_line_map = {}
    # Plot the initial positions of the vehicles and obstacles.
    for vehicle_id, vehicle in enumerate(vehicles):
        x, y = vehicle.dynamics.initial_state[:2]
        line = ax.plot(x, y, marker="o", color=vehicle_colors[vehicle_id])[0]
        vehicle_line_map[vehicle_id] = line

    def update(frame):
        for vehicle_id, vehicle in enumerate(vehicles):
            state_trajectory = vst_map[vehicle_id]
            x, y = state_trajectory[frame][:2]
            vehicle_line_map[vehicle_id].set_xdata(x)
            vehicle_line_map[vehicle_id].set_ydata(y)

    _ = anim.FuncAnimation(fig=fig, func=update, frames=nt, interval=100, repeat=True)

    plt.show()
