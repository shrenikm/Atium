from typing import Dict


def state_variable_str_from_ids(
    vehicle_id: int,
    time_step_id: int,
    state_id: int,
) -> str:
    return f"s_{vehicle_id}_{time_step_id}_{state_id}"


def state_slack_variable_str_from_ids(
    vehicle_id: int,
    time_step_id: int,
    state_id: int,
) -> str:
    return f"w_{vehicle_id}_{time_step_id}_{state_id}"


def control_variable_str_from_ids(
    vehicle_id: int,
    time_step_id: int,
    control_id: int,
) -> str:
    return f"u_{vehicle_id}_{time_step_id}_{control_id}"


def control_slack_variable_str_from_ids(
    vehicle_id: int,
    time_step_id: int,
    control_id: int,
) -> str:
    return f"v_{vehicle_id}_{time_step_id}_{control_id}"


def vehicle_obstacle_collision_binary_slack_variable_str_from_ids(
    vehicle_id: int,
    obstacle_id: int,
    time_step_id: int,
    var_id: int,
) -> str:
    assert 0 <= var_id < 4
    return f"t_{vehicle_id}_{obstacle_id}_{time_step_id}_{var_id}"


def vehicle_vehicle_collision_binary_slack_variable_str_from_ids(
    current_vehicle_id: int,
    other_vehicle_id: int,
    time_step_id: int,
    var_id: int,
) -> str:
    assert 0 <= var_id < 4
    return f"b_{current_vehicle_id}_{other_vehicle_id}_{time_step_id}_{var_id}"


def state_slack_constraint_var_from_var_strs(
    state_var_str: str,
    state_slack_var_str: str,
    constraint_id: int,
) -> str:
    return f"c_ss_{state_var_str}_{state_slack_var_str}_{constraint_id}"


def control_slack_constraint_var_from_var_strs(
    control_var_str: str,
    control_slack_var_str: str,
    constraint_id: int,
) -> str:
    return f"c_cs_{control_var_str}_{control_slack_var_str}_{constraint_id}"


def state_transition_constraint_var_from_var_strs(
    vehicle_id: int,
    current_time_step_id: int,
    next_time_step_id: int,
    constraint_id: int,
) -> str:
    return (
        f"c_st_{vehicle_id}_{current_time_step_id}_{next_time_step_id}_{constraint_id}"
    )


def vehicle_obstacle_collision_constraint_var_from_var_strs(
    state_var_str: str,
    binary_var_str: str,
) -> str:
    return f"c_voc_{state_var_str}_{binary_var_str}"


def vehicle_obstacle_collision_binary_constraint_var_from_ids(
    vehicle_id: int,
    obstacle_id: int,
    time_step_id: int,
) -> str:
    return f"c_vocb_{vehicle_id}_{obstacle_id}_{time_step_id}"  # Sum of t's constraint


def vehicle_vehicle_collision_constraint_var_from_var_strs(
    current_state_var_str: str,
    other_state_var_str: str,
    binary_var_str: str,
) -> str:
    return f"c_vvc_{current_state_var_str}_{other_state_var_str}_{binary_var_str}"


def vehicle_vehicle_collision_binary_constraint_var_from_ids(
    current_vehicle_id: int,
    other_vehicle_id: int,
    time_step_id: int,
) -> str:
    return f"c_vvcb_{current_vehicle_id}_{other_vehicle_id}_{time_step_id}"  # Sum of b's constraint


def assert_uniqueness_and_update_mvmip_map(
    mvmip_map_to_be_added: Dict,
    mvmip_map: Dict,
) -> None:
    """
    The map is going to be from a str value to some MVMIP construct (variable/constraint/etc).
    Adding the contents of mvmip_map_to_be_added into mvmip_map, but while first making sure
    that they keys are unique and don't already exist in mvmip_map as we don't want to
    overwrite constraints/variables.
    """
    for mvmip_key in mvmip_map_to_be_added.keys():
        assert mvmip_key not in mvmip_map
    mvmip_map.update(mvmip_map_to_be_added)
