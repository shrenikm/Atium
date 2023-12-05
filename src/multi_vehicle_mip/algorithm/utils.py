def state_variable_str_from_ids(
    vehicle_id: int,
    time_step_id: int,
    state_id: int,
) -> str:
    return f"s_{vehicle_id}{time_step_id}{state_id}"


def state_slack_variable_str_from_ids(
    vehicle_id: int,
    time_step_id: int,
    state_id: int,
) -> str:
    return f"w_{vehicle_id}{time_step_id}{state_id}"


def control_variable_str_from_ids(
    vehicle_id: int,
    time_step_id: int,
    control_id: int,
) -> str:
    return f"u_{vehicle_id}{time_step_id}{control_id}"


def control_slack_variable_str_from_ids(
    vehicle_id: int,
    time_step_id: int,
    control_id: int,
) -> str:
    return f"v_{vehicle_id}{time_step_id}{control_id}"
