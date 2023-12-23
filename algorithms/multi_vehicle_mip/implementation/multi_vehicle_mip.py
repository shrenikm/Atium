from typing import Optional, Sequence
import time

from ortools.linear_solver import pywraplp
from algorithms.multi_vehicle_mip.implementation.constraints import (
    construct_constraints_for_mvmip,
)
from algorithms.multi_vehicle_mip.implementation.definitions import (
    MVMIPOptimizationParams,
    MVMIPResult,
    MVMIPVehicle,
    MVMIPObstacle,
)
from algorithms.multi_vehicle_mip.implementation.objective import (
    construct_objective_for_mvmip,
    mvmip_result_from_solver,
)
from algorithms.multi_vehicle_mip.implementation.variables import (
    construct_variables_for_mvmip,
)


def solve_mvmip(
    mvmip_params: MVMIPOptimizationParams,
    vehicles: Sequence[MVMIPVehicle],
    obstacles: Sequence[MVMIPObstacle],
) -> Optional[MVMIPResult]:

    # TODO: Check validity of vehicles and obstacles.

    pre_setup_time = time.perf_counter()
    solver: pywraplp.Solver = pywraplp.Solver.CreateSolver("SCIP")
    assert solver is not None, "Solver could not be created."

    # Variables
    vars_map = construct_variables_for_mvmip(
        solver=solver,
        mvmip_params=mvmip_params,
        vehicles=vehicles,
        obstacles=obstacles,
    )
    assert len(vars_map) == solver.NumVariables()
    for var_name, var in vars_map.items():
        assert var_name == var.name()

    # Constraints
    cons_map = construct_constraints_for_mvmip(
        solver=solver,
        mvmip_params=mvmip_params,
        vehicles=vehicles,
        obstacles=obstacles,
    )
    assert len(cons_map) == solver.NumConstraints()
    for cons_name, cons in cons_map.items():
        assert cons_name == cons.name()

    # Objective
    objective = construct_objective_for_mvmip(
        solver=solver,
        mvmip_params=mvmip_params,
        vehicles=vehicles,
    )
    post_setup_time = time.perf_counter()

    pre_solve_time = time.perf_counter()
    status = solver.Solve()
    post_solve_time = time.perf_counter()

    if status == solver.OPTIMAL:
        print("Optimal solution for MVMIP found!")
        return mvmip_result_from_solver(
            solver=solver,
            mvmip_params=mvmip_params,
            vehicles=vehicles,
            obstacles=obstacles,
            solver_setup_time_s=post_setup_time - pre_setup_time,
            solver_solve_time_s=post_solve_time - pre_solve_time,
        )
    else:
        print("Optimal solution for MVMIP could not be found :(")
        return None
