import numpy as np

from atium.core.constructs.environment_map import EnvironmentLabels, EnvironmentMap2D
from atium.core.utils.custom_types import PolygonXYArray
from atium.core.utils.geometry_utils import construct_rectangle_polygon, densify_polygon
from atium.implementations.unito.src.unito_utils import UnitoFinalStateInputs, UnitoInitialStateInputs, UnitoInputs


def get_scenario_footprint() -> PolygonXYArray:
    footprint_spacing = 0.5
    footprint_size_xy = (1.0, 0.4)
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
    footprint = get_scenario_footprint()
    emap2d = EnvironmentMap2D.from_empty(
        size_xy=(5.0, 5.0),
        resolution=0.1,
    )
    emap2d.add_rectangular_obstacle(
        center_xy=(2.5, 2.5),
        size_xy=(0.1, 1.0),
        label=EnvironmentLabels.STATIC_OBSTACLE,
    )

    obstacle_clearance = 0.2
    initial_state_inputs = UnitoInitialStateInputs(
        initial_ms_map={
            0: np.array([0.0, 0.0]),
        },
        initial_xy=np.array([1.0, 2.0]),
    )
    final_state_inputs = UnitoFinalStateInputs(
        final_ms_map={
            # 0: np.array([0.0, 0.0]),
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
