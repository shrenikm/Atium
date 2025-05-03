import numpy as np

from atium.implementations.unito.scenarios import scenario1
from atium.implementations.unito.src.unito import Unito
from atium.implementations.unito.src.unito_utils import UnitoInputs, UnitoParams
from atium.implementations.unito.src.unito_variable_manager import UnitoVariableManager


def run_unito_scenario(
    unito_params: UnitoParams,
    unito_inputs: UnitoInputs,
    debug_solver: bool,
    visualize_solution: bool,
) -> None:
    manager = UnitoVariableManager(params=unito_params)
    unito = Unito(manager=manager)

    unito.setup_optimization_program(inputs=unito_inputs)
    initial_guess = unito.compute_initial_guess(inputs=unito_inputs)
    unito.solve(
        inputs=unito_inputs,
        initial_guess=initial_guess,
        debug_solver=debug_solver,
        visualize_solution=visualize_solution,
    )


if __name__ == "__main__":
    unito_params = UnitoParams(
        h=3,
        M=3,
        n=4,
        epsilon_t=0,
        W=1e-1 * np.ones((2, 2), dtype=np.float64),
    )
    unito_inputs = scenario1()
    debug_solver = True
    visualize_solution = True

    run_unito_scenario(
        unito_params=unito_params,
        unito_inputs=unito_inputs,
        debug_solver=debug_solver,
        visualize_solution=visualize_solution,
    )
