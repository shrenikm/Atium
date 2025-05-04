import numpy as np

from atium.core.constructs.environment_map import EnvironmentLabels, EnvironmentMap2D
from atium.core.utils.custom_types import PolygonXYArray, SizeXY
from atium.core.utils.geometry_utils import construct_rectangle_polygon, densify_polygon
from atium.implementations.unito.src.unito_utils import (
    UnitoFinalStateInputs,
    UnitoInitialStateInputs,
    UnitoInputs,
    UnitoMotionState,
)


def get_scenario_footprint(footprint_size_xy: SizeXY) -> PolygonXYArray:
    footprint_spacing = 0.2
    return densify_polygon(
        polygon=construct_rectangle_polygon(
            center_xy=(0.0, 0.0),
            size_xy=footprint_size_xy,
        ),
        spacing=footprint_spacing,
    )


def scenario1() -> UnitoInputs:
    """
    Scenario 1: Planning around a single rectangular obstacle.
    """
    footprint_size_xy = (1.0, 0.4)
    footprint = get_scenario_footprint(footprint_size_xy=footprint_size_xy)
    emap2d = EnvironmentMap2D.from_empty(
        size_xy=(5.0, 5.0),
        resolution=0.1,
    )
    # Add rectangular obstacle
    obstacle_length = 1.0
    obstacle_width = 0.1
    emap2d.add_rectangular_obstacle(
        center_xy=(2.5, 2.5),
        size_xy=(obstacle_width, obstacle_length),
        label=EnvironmentLabels.STATIC_OBSTACLE,
    )

    obstacle_clearance = 0.2
    initial_state_inputs = UnitoInitialStateInputs(
        initial_ms_map={
            0: UnitoMotionState(theta=0.0, s=0.0),
        },
        initial_xy=np.array([1.0, 2.0]),
    )
    final_state_inputs = UnitoFinalStateInputs(
        final_ms_map={
            0: UnitoMotionState(theta=0.0, s=0.0),
        },
        final_xy=np.array([4.0, 2.0]),
    )
    return UnitoInputs(
        footprint=footprint,
        emap2d=emap2d,
        obstacle_clearance=obstacle_clearance,
        initial_state_inputs=initial_state_inputs,
        final_state_inputs=final_state_inputs,
    )


def scenario2() -> UnitoInputs:
    """
    Scenario 2: Planning throw a narrow corridor
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
    initial_state_inputs = UnitoInitialStateInputs(
        initial_ms_map={
            0: UnitoMotionState(theta=0.0, s=0.0),
        },
        initial_xy=np.array([1.0, 2.5]),
    )
    final_state_inputs = UnitoFinalStateInputs(
        final_ms_map={
            0: UnitoMotionState(theta=0.0, s=0.0),
        },
        final_xy=np.array([4.0, 2.5]),
    )
    return UnitoInputs(
        footprint=footprint,
        emap2d=emap2d,
        obstacle_clearance=obstacle_clearance,
        initial_state_inputs=initial_state_inputs,
        final_state_inputs=final_state_inputs,
    )


def scenario3() -> UnitoInputs:
    """
    Scenario 3: Tight turn into a narrow corridor
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
    initial_state_inputs = UnitoInitialStateInputs(
        initial_ms_map={
            0: UnitoMotionState(theta=0.0, s=0.0),
        },
        initial_xy=np.array([1.0, 1.0]),
    )
    final_state_inputs = UnitoFinalStateInputs(
        final_ms_map={
            0: UnitoMotionState(theta=np.pi / 2.0, s=0.0),
        },
        final_xy=np.array([2.5, wall_y]),
    )
    return UnitoInputs(
        footprint=footprint,
        emap2d=emap2d,
        obstacle_clearance=obstacle_clearance,
        initial_state_inputs=initial_state_inputs,
        final_state_inputs=final_state_inputs,
    )
