import numpy as np

from atium.core.utils.custom_types import StateVector
from atium.implementations.multi_vehicle_mip.src.definitions import MVMIPVehicleDynamics


def create_standard_omni_vehicle_dynamics(
    initial_state: StateVector,
    final_state: StateVector,
    clearance_m: float,
    dt: float,
) -> MVMIPVehicleDynamics:
    assert initial_state.ndim == 1 and final_state.ndim == 1
    assert initial_state.size == final_state.size

    nx = initial_state.size
    a_matrix = np.eye(nx, dtype=np.float64)
    b_matrix = dt * np.eye(nx, dtype=np.float64)

    return MVMIPVehicleDynamics(
        a_matrix=a_matrix,
        b_matrix=b_matrix,
        initial_state=initial_state,
        final_state=final_state,
        clearance_m=clearance_m,
    )
