"""
Runs the cartpole simulation with a controller that outputs random values between its limits.
"""
import numpy as np

from common.constants import ACC_GRAVITY
from common.control.basic_controllers import (
    ConstantController,
    UniformPDFController,
    ZeroController,
)
from common.dynamics.cartpole_dyn import CartpoleParams
from common.dynamics.utils import ControlInputVectorLimits, StateVectorLimits
from common.simulation.envs.cartpole_env import CartpoleSiliconEnv
from common.simulation.integrators.state_integrators import StateIntegratorType

if __name__ == "__main__":

    dt = 0.01
    initial_state = np.zeros(4)
    initial_control_input = np.zeros(1)

    x_min, x_max = -4.0, 4.0
    f_x_min, f_x_max = -50.0, 50.0
    state_lower = np.array([x_min, -np.inf, -np.inf, -np.inf], dtype=np.float64)
    state_upper = np.array([x_max, np.inf, np.inf, np.inf], dtype=np.float64)
    control_input_lower = np.array([f_x_min], dtype=np.float64)
    control_input_upper = np.array([f_x_max], dtype=np.float64)
    state_limits = StateVectorLimits(
        lower=state_lower,
        upper=state_upper,
    )
    control_input_limits = ControlInputVectorLimits(
        lower=control_input_lower,
        upper=control_input_upper,
    )

    params = CartpoleParams(
        m_c=1.0,
        m_p=0.5,
        l=1.0,
        g=ACC_GRAVITY,
    )
    cartpole_env = CartpoleSiliconEnv.from_params(
        initial_state=initial_state,
        initial_control_input=initial_control_input,
        state_limits=state_limits,
        control_input_limits=control_input_limits,
        params=params,
    )

    # Different controllers.

    # Passive controller (Note: no friction)
    zero_controller = ZeroController(size=1)
    # Constant controller
    constant_controller = ConstantController(control_input=np.array([0.5]))
    # Random controller
    uniform_controller = UniformPDFController(
        control_input_limits=ControlInputVectorLimits(
            lower=np.array([f_x_min]),
            upper=np.array([f_x_max]),
        ),
    )

    controllers = [zero_controller, constant_controller, uniform_controller]

    controller_id = 2
    assert controller_id < len(controllers)
    controller = controllers[controller_id]

    cartpole_env.step_simulate(
        controller=controller,
        dt=dt,
        state_integrator_type=StateIntegratorType.EXPLICIT_EULER,
    )
