import numpy as np
import time
from algorithms.multi_vehicle_mip.implementation.definitions import (
    MVMIPOptimizationParams,
    MVMIPRectangleObstacle,
    MVMIPVehicleDynamics,
    MVMIPVehicleOptimizationParams,
    MVMIPVehicle,
)
from algorithms.multi_vehicle_mip.implementation.multi_vehicle_mip import solve_mvmip

from algorithms.multi_vehicle_mip.standard_vehicles import (
    create_standard_omni_vehicle_dynamics,
)


if __name__ == "__main__":

    num_time_steps = 50
    dt = 0.1
    world_size = 10.0
    control_max = 2.0
    M = 1e6

    # Optimization params
    mvmip_params = MVMIPOptimizationParams(
        num_time_steps=num_time_steps,
        dt=dt,
        M=M,
    )

    # Setup vehicles
    initial_state = np.array([1.0, 1.0], dtype=np.float64)
    final_state = np.array([8.0, 8.0], dtype=np.float64)
    clearance_m = 0.5

    dynamics: MVMIPVehicleDynamics = create_standard_omni_vehicle_dynamics(
        initial_state=initial_state,
        final_state=final_state,
        clearance_m=clearance_m,
        dt=mvmip_params.dt,
    )
    optimization_params = MVMIPVehicleOptimizationParams(
        q_cost_vector=np.array([1.0, 1.0], dtype=np.float64),
        r_cost_vector=np.array([1.0, 1.0], dtype=np.float64),
        p_cost_vector=np.array([1.0, 1.0], dtype=np.float64),
        state_min=np.array([0.0, 0.0], dtype=np.float64),
        state_max=np.array([world_size, world_size], dtype=np.float64),
        control_min=np.array([-control_max, -control_max], dtype=np.float64),
        control_max=np.array([control_max, control_max], dtype=np.float64),
    )
    vehicle = MVMIPVehicle(
        dynamics=dynamics,
        optimization_params=optimization_params,
    )
    vehicles = [
        vehicle,
    ]

    # Setup obstacles
    obstacle = MVMIPRectangleObstacle(
        initial_center_xy=np.array(
            [world_size / 2.0, world_size / 2.0], dtype=np.float64
        ),
        size_xy_m=np.array([2.0, 2.0], dtype=np.float64),
        velocities_xy_mps=np.array([0.0, 0.0], dtype=np.float64),
        clearance_m=0.2,
        # TODO: Figure out another way to access these in the obstacle
        num_time_steps=num_time_steps,
        dt=dt,
    )
    obstacles = [
        obstacle,
    ]

    start_time = time.perf_counter()
    solve_mvmip(
        mvmip_params=mvmip_params,
        vehicles=vehicles,
        obstacles=obstacles,
    )
    end_time = time.perf_counter()
    print(f"Time to solve MVMIP: {end_time - start_time} seconds")
