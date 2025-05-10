import numpy as np

import atium.experiments.runito.runito_scenarios as sc
from atium.experiments.runito.src.runito import Runito
from atium.experiments.runito.src.runito_utils import RunitoInputs, RunitoParams
from atium.experiments.runito.src.runito_variable_manager import RunitoVariableManager


def run_runito_scenario(
    runito_params: RunitoParams,
    runito_inputs: RunitoInputs,
    debug_solver: bool,
    visualize_solution: bool,
) -> None:
    manager = RunitoVariableManager(params=runito_params)
    runito = Runito(manager=manager)

    runito.setup_optimization_program(inputs=runito_inputs)
    initial_guess = runito.compute_initial_guess(inputs=runito_inputs)
    runito.solve(
        inputs=runito_inputs,
        initial_guess=initial_guess,
        debug_solver=debug_solver,
        visualize_solution=visualize_solution,
    )


if __name__ == "__main__":
    runito_params = RunitoParams(
        h=3,
        M=3,
        n=4,
        epsilon_t=0,
        W=1e-2 * np.eye(3, dtype=np.float64),
    )
    runito_inputs = sc.scenario2()
    debug_solver = True
    visualize_solution = True

    run_runito_scenario(
        runito_params=runito_params,
        runito_inputs=runito_inputs,
        debug_solver=debug_solver,
        visualize_solution=visualize_solution,
    )
