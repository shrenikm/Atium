import yaml

from algorithms.multi_vehicle_mip.implementation.multi_vehicle_mip import solve_mvmip
from algorithms.multi_vehicle_mip.implementation.setups.utils import (
    animation_params_from_setup_yaml_dict,
    get_full_path_of_setup_yaml,
    mvmip_params_from_setup_yaml_dict,
    obstacles_from_setup_yaml_dict,
    vehicles_from_setup_yaml_dict,
)
from algorithms.multi_vehicle_mip.implementation.visualization import (
    visualize_mvmip_result,
)


if __name__ == "__main__":

    setup_yaml_filename = "mvmip_setup2.yaml"

    setup_file_path = get_full_path_of_setup_yaml(
        setup_yaml_filename=setup_yaml_filename
    )
    with open(setup_file_path, "r") as file:
        setup_yaml_dict = yaml.safe_load(file)

    mvmip_params = mvmip_params_from_setup_yaml_dict(
        setup_yaml_dict=setup_yaml_dict,
    )
    vehicles = vehicles_from_setup_yaml_dict(
        setup_yaml_dict=setup_yaml_dict,
    )
    obstacles = obstacles_from_setup_yaml_dict(
        setup_yaml_dict=setup_yaml_dict,
    )
    animation_params = animation_params_from_setup_yaml_dict(
        setup_yaml_dict=setup_yaml_dict,
    )

    result = solve_mvmip(
        mvmip_params=mvmip_params,
        vehicles=vehicles,
        obstacles=obstacles,
    )

    print(result)
    if result is not None:
        print(f"MVMIP setup time: {result.solver_setup_time_s} s")
        print(f"MVMIP solve time: {result.solver_solving_time_s} s")
        visualize_mvmip_result(
            mvmip_result=result,
            animation_params=animation_params,
        )
