import os
from typing import Sequence

import attr

from algorithms.multi_vehicle_mip.implementation.custom_types import SetupYamlDict
from algorithms.multi_vehicle_mip.implementation.definitions import (
    MVMIPObstacle,
    MVMIPOptimizationParams,
    MVMIPRectangleObstacle,
    MVMIPVehicle,
    MVMIPVehicleDynamics,
    MVMIPVehicleOptimizationParams,
)
from algorithms.multi_vehicle_mip.implementation.visualization import MVMIPAnimationParams
from common.custom_types import FileName, FilePath

SETUP_YAML_MVMIP_PARAMS_KEY = "mvmip_params"
SETUP_YAML_MVMIP_PARAMS_DT_KEY = "dt"
SETUP_YAML_VEHICLES_KEY = "vehicles"
SETUP_YAML_VEHICLE_DYNAMICS_KEY = "dynamics"
SETUP_YAML_VEHICLE_DYNAMICS_A_MATRIX_KEY = "a_matrix"
SETUP_YAML_VEHICLE_DYNAMICS_B_MATRIX_KEY = "b_matrix"
SETUP_YAML_VEHICLE_OPTIMIZATION_PARAMS_KEY = "optimization_params"
SETUP_YAML_OBSTACLES_KEY = "obstacles"
SETUP_YAML_ANIMATION_PARAMS_KEY = "animation_params"


def get_full_path_of_setup_yaml(
    setup_yaml_filename: FileName,
) -> FilePath:
    directory_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(directory_path, setup_yaml_filename)


def mvmip_params_from_setup_yaml_dict(
    setup_yaml_dict: SetupYamlDict,
) -> MVMIPOptimizationParams:
    return MVMIPOptimizationParams(
        **setup_yaml_dict[SETUP_YAML_MVMIP_PARAMS_KEY],
    )


def vehicles_from_setup_yaml_dict(
    setup_yaml_dict: SetupYamlDict,
) -> Sequence[MVMIPVehicle]:
    dt = setup_yaml_dict[SETUP_YAML_MVMIP_PARAMS_KEY][SETUP_YAML_MVMIP_PARAMS_DT_KEY]

    vehicles_dict = setup_yaml_dict[SETUP_YAML_VEHICLES_KEY]
    vehicles = []

    for vehicle_dict in vehicles_dict.values():
        dynamics_dict = vehicle_dict[SETUP_YAML_VEHICLE_DYNAMICS_KEY]
        optimization_params_dict = vehicle_dict[
            SETUP_YAML_VEHICLE_OPTIMIZATION_PARAMS_KEY
        ]

        dynamics = MVMIPVehicleDynamics(**dynamics_dict)
        # Incorporate dt into the b_matrix
        dynamics = attr.evolve(
            dynamics,
            b_matrix=dt * dynamics.b_matrix,
        )
        optimization_params = MVMIPVehicleOptimizationParams(**optimization_params_dict)

        vehicles.append(
            MVMIPVehicle(
                dynamics=dynamics,
                optimization_params=optimization_params,
            ),
        )
    return vehicles


def obstacles_from_setup_yaml_dict(
    setup_yaml_dict: SetupYamlDict,
) -> Sequence[MVMIPObstacle]:

    obstacles_dict = setup_yaml_dict[SETUP_YAML_OBSTACLES_KEY]
    obstacles = []

    for obstacle_dict in obstacles_dict.values():
        try:
            obstacle = MVMIPRectangleObstacle(**obstacle_dict)
        except TypeError as e:
            raise NotImplementedError(
                f"Only rectangular obstacles have been implemented so far."
            ) from e
        obstacles.append(obstacle)

    return obstacles


def animation_params_from_setup_yaml_dict(
    setup_yaml_dict: SetupYamlDict,
) -> MVMIPAnimationParams:

    animation_params_dict = setup_yaml_dict[SETUP_YAML_ANIMATION_PARAMS_KEY]
    return MVMIPAnimationParams(**animation_params_dict)
