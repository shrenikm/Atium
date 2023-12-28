import numpy as np
from common.control.basic_controllers import UniformPDFController
from common.custom_types import ControlInputVectorLimits, StateVectorLimits

from common.simulation.envs.cartpole_env import CartpoleSiliconEnv

if __name__ == "__main__":

    dt = 0.01
    initial_state = np.zeros(4)
    initial_control_input = np.zeros(1)

    x_min, x_max = -5.0, 5.0
    f_x_min, f_x_max = -10.0, 10.0
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

    cartpole_env = CartpoleSiliconEnv(
        state=initial_state,
        control_input=initial_control_input,
        state_limits=state_limits,
        control_input_limits=control_input_limits,
    )

    controller = UniformPDFController(
        control_input_limits=ControlInputVectorLimits(
            lower=np.array([-0.5]),
            upper=np.array([0.5]),
        ),
    )

    cartpole_env.step_simulate(
        controller=controller,
        dt=dt,
    )
