import numpy as np

from atium.core.definitions.concrete_states import Pose2D, Velocity2D
from atium.core.definitions.environment_map import EnvironmentLabels, EnvironmentMap2D
from atium.core.utils.custom_types import PolygonXYArray, SizeXY
from atium.core.utils.geometry_utils import construct_rectangle_polygon, densify_polygon
from atium.experiments.runito.src.runito_utils import RunitoFinalStateInputs, RunitoInitialStateInputs, RunitoInputs


def get_scenario_footprint(footprint_size_xy: SizeXY) -> PolygonXYArray:
    footprint_spacing = 0.2
    return densify_polygon(
        polygon=construct_rectangle_polygon(
            center_xy=(0.0, 0.0),
            size_xy=footprint_size_xy,
        ),
        spacing=footprint_spacing,
    )


def get_scenario_velocity_limits() -> tuple[Velocity2D, Velocity2D]:
    v_min, v_max = 0.0, 10.0
    w_min, w_max = -10.0, 10.0
    return (
        Velocity2D(linear=v_min, angular=w_min),
        Velocity2D(linear=v_max, angular=w_max),
    )


def scenario1() -> RunitoInputs:
    """
    Scenario 1: Planning in a space with no obstacles.
    """
    footprint_size_xy = (1.0, 0.4)
    footprint = get_scenario_footprint(footprint_size_xy=footprint_size_xy)
    emap2d = EnvironmentMap2D.from_empty(
        size_xy=(5.0, 5.0),
        resolution=0.1,
    )

    obstacle_clearance = 0.0
    initial_state_inputs = RunitoInitialStateInputs(
        initial_pose=Pose2D.from_vector(np.array([1.0, 2.0, 0.0])),
        initial_velocity=Velocity2D.from_vector(np.zeros(2)),
    )
    final_state_inputs = RunitoFinalStateInputs(
        final_pose=Pose2D.from_vector(np.array([3.0, 2.0, 0.0])),
    )
    lower_velocity_limits, upper_velocity_limits = get_scenario_velocity_limits()
    return RunitoInputs(
        footprint=footprint,
        emap2d=emap2d,
        obstacle_clearance=obstacle_clearance,
        lower_velocity_limits=lower_velocity_limits,
        upper_velocity_limits=upper_velocity_limits,
        initial_state_inputs=initial_state_inputs,
        final_state_inputs=final_state_inputs,
    )


def scenario2() -> RunitoInputs:
    """
    Scenario 2: Planning around a single rectangular obstacle.
    """
    footprint_size_xy = (1.0, 0.4)
    footprint = get_scenario_footprint(footprint_size_xy=footprint_size_xy)
    emap2d = EnvironmentMap2D.from_empty(
        size_xy=(5.0, 5.0),
        resolution=0.1,
    )
    # Add rectangular obstacle
    obstacle_length = 1.0
    obstacle_width = 0.2
    emap2d.add_rectangular_obstacle(
        center_xy=(2.5, 2.5),
        size_xy=(obstacle_width, obstacle_length),
        label=EnvironmentLabels.STATIC_OBSTACLE,
    )

    obstacle_clearance = 0.2
    robot_y = 2.
    initial_state_inputs = RunitoInitialStateInputs(
        initial_pose=Pose2D.from_vector(np.array([1.0, robot_y, 0.0])),
        initial_velocity=Velocity2D.from_vector(np.zeros(2)),
    )
    final_state_inputs = RunitoFinalStateInputs(
        final_pose=Pose2D.from_vector(np.array([4.0, robot_y, 0.0])),
    )
    lower_velocity_limits, upper_velocity_limits = get_scenario_velocity_limits()
    return RunitoInputs(
        footprint=footprint,
        emap2d=emap2d,
        obstacle_clearance=obstacle_clearance,
        lower_velocity_limits=lower_velocity_limits,
        upper_velocity_limits=upper_velocity_limits,
        initial_state_inputs=initial_state_inputs,
        final_state_inputs=final_state_inputs,
    )


def scenario3() -> RunitoInputs:
    """
    Scenario 3: Planning throw a narrow corridor
    """
    footprint_size_xy = (1.0, 0.4)
    footprint = get_scenario_footprint(footprint_size_xy=footprint_size_xy)
    emap2d = EnvironmentMap2D.from_empty(
        size_xy=(5.0, 5.0),
        resolution=0.01,
    )
    # Add walls.
    wall_clearance = 0.05
    wall_length = 4.0
    wall_thickness = 0.2
    offset = 0.5 * (footprint_size_xy[1] + wall_thickness) + wall_clearance
    emap2d.add_rectangular_obstacle(
        center_xy=(2.5, 2.5 + offset),
        size_xy=(wall_length, wall_thickness),
        label=EnvironmentLabels.STATIC_OBSTACLE,
    )
    emap2d.add_rectangular_obstacle(
        center_xy=(2.5, 2.5 - offset),
        size_xy=(wall_length, wall_thickness),
        label=EnvironmentLabels.STATIC_OBSTACLE,
    )

    obstacle_clearance = 0.0
    initial_state_inputs = RunitoInitialStateInputs(
        initial_pose=Pose2D.from_vector(np.array([1.0, 2.5, 0.0])),
        initial_velocity=Velocity2D.from_vector(np.zeros(2)),
    )
    final_state_inputs = RunitoFinalStateInputs(
        final_pose=Pose2D.from_vector(np.array([4.0, 2.5, 0.0])),
    )
    lower_velocity_limits, upper_velocity_limits = get_scenario_velocity_limits()
    return RunitoInputs(
        footprint=footprint,
        emap2d=emap2d,
        obstacle_clearance=obstacle_clearance,
        lower_velocity_limits=lower_velocity_limits,
        upper_velocity_limits=upper_velocity_limits,
        initial_state_inputs=initial_state_inputs,
        final_state_inputs=final_state_inputs,
    )


def scenario4() -> RunitoInputs:
    """
    Scenario 4: Tight turn into a narrow corridor
    """
    footprint_size_xy = (1.0, 0.4)
    footprint = get_scenario_footprint(footprint_size_xy=footprint_size_xy)
    emap2d = EnvironmentMap2D.from_empty(
        size_xy=(5.0, 5.0),
        resolution=0.01,
    )
    # Add walls.
    wall_clearance = 0.1
    wall_length = 3.0
    wall_thickness = 0.2
    wall_y = 3.0
    offset = 0.5 * (footprint_size_xy[1] + wall_thickness) + wall_clearance
    emap2d.add_rectangular_obstacle(
        center_xy=(2.5 - offset, wall_y),
        size_xy=(wall_thickness, wall_length),
        label=EnvironmentLabels.STATIC_OBSTACLE,
    )
    emap2d.add_rectangular_obstacle(
        center_xy=(2.5 + offset, wall_y),
        size_xy=(wall_thickness, wall_length),
        label=EnvironmentLabels.STATIC_OBSTACLE,
    )

    obstacle_clearance = 0.0
    initial_state_inputs = RunitoInitialStateInputs(
        initial_pose=Pose2D.from_vector(np.array([1.0, 1.0, 0.0])),
        initial_velocity=Velocity2D.from_vector(np.zeros(2)),
    )
    final_state_inputs = RunitoFinalStateInputs(
        final_pose=Pose2D.from_vector(np.array([2.5, wall_y, np.pi / 2.0])),
    )
    lower_velocity_limits, upper_velocity_limits = get_scenario_velocity_limits()
    return RunitoInputs(
        footprint=footprint,
        emap2d=emap2d,
        obstacle_clearance=obstacle_clearance,
        lower_velocity_limits=lower_velocity_limits,
        upper_velocity_limits=upper_velocity_limits,
        initial_state_inputs=initial_state_inputs,
        final_state_inputs=final_state_inputs,
    )


def scenario5() -> RunitoInputs:
    """
    Scenario 5: U-turn in a narrow corridor where it needs to make the turn outside the corridor
    """
    footprint_size_xy = (1.0, 0.4)
    footprint = get_scenario_footprint(footprint_size_xy=footprint_size_xy)
    emap2d = EnvironmentMap2D.from_empty(
        size_xy=(5.0, 5.0),
        resolution=0.01,
    )
    # Add walls.
    wall_clearance = 0.1
    wall_length = 3.0
    wall_thickness = 0.5
    offset = 0.5 * (footprint_size_xy[1] + wall_thickness) + wall_clearance
    emap2d.add_rectangular_obstacle(
        center_xy=(1.5, 2.5 + offset),
        size_xy=(wall_length, wall_thickness),
        label=EnvironmentLabels.STATIC_OBSTACLE,
    )
    emap2d.add_rectangular_obstacle(
        center_xy=(1.5, 2.5 - offset),
        size_xy=(wall_length, wall_thickness),
        label=EnvironmentLabels.STATIC_OBSTACLE,
    )

    obstacle_clearance = 0.0001
    initial_state_inputs = RunitoInitialStateInputs(
        initial_pose=Pose2D.from_vector(np.array([2.5, 2.5, 0.0])),
        initial_velocity=Velocity2D.from_vector(np.zeros(2)),
    )
    final_state_inputs = RunitoFinalStateInputs(
        final_pose=Pose2D.from_vector(np.array([1.5, 2.5, np.pi])),
    )
    lower_velocity_limits, upper_velocity_limits = get_scenario_velocity_limits()
    return RunitoInputs(
        footprint=footprint,
        emap2d=emap2d,
        obstacle_clearance=obstacle_clearance,
        lower_velocity_limits=lower_velocity_limits,
        upper_velocity_limits=upper_velocity_limits,
        initial_state_inputs=initial_state_inputs,
        final_state_inputs=final_state_inputs,
    )
