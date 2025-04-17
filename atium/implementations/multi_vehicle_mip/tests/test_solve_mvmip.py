import pytest
import yaml

from atium.implementations.multi_vehicle_mip.src.definitions import MVMIPResult
from atium.implementations.multi_vehicle_mip.src.multi_vehicle_mip import solve_mvmip
from atium.implementations.multi_vehicle_mip.src.setups.utils import (
    get_full_path_of_setup_yaml,
    mvmip_params_from_setup_yaml_dict,
    obstacles_from_setup_yaml_dict,
    vehicles_from_setup_yaml_dict,
)

MVMIP_SETUPS_TO_TEST = [
    "mvmip_setup1.yaml",
    "mvmip_setup7.yaml",
]


@pytest.mark.parametrize("setup_yaml_filename", MVMIP_SETUPS_TO_TEST)
def test_solve_mvmip_setup_through_yaml(setup_yaml_filename: str) -> None:
    setup_file_path = get_full_path_of_setup_yaml(setup_yaml_filename=setup_yaml_filename)
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

    result = solve_mvmip(
        mvmip_params=mvmip_params,
        vehicles=vehicles,
        obstacles=obstacles,
    )
    assert isinstance(result, MVMIPResult)


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
