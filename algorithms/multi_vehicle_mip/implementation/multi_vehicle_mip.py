from typing import Sequence

from ortools.linear_solver import pywraplp
from algorithms.multi_vehicle_mip.implementation.constraints import (
    construct_constraints_for_mvmip,
)
from algorithms.multi_vehicle_mip.implementation.definitions import (
    MVMIPOptimizationParams,
    MVMIPVehicle,
    MVMIPObstacle,
)
from algorithms.multi_vehicle_mip.implementation.variables import (
    construct_variables_for_mvmip,
)
from common.custom_types import StateTrajectoryArray


def solve_mvmip(
    mvmip_params: MVMIPOptimizationParams,
    vehicles: Sequence[MVMIPVehicle],
    obstacles: Sequence[MVMIPObstacle],
) -> Sequence[StateTrajectoryArray]:

    # TODO: Check validity of vehicles and obstacles.

    solver: pywraplp.Solver = pywraplp.Solver.CreateSolver("SCIP")
    assert solver is not None, "Solver could not be created."

    vars_map = construct_variables_for_mvmip(
        solver=solver,
        mvmip_params=mvmip_params,
        vehicles=vehicles,
        obstacles=obstacles,
    )
    assert len(vars_map) == solver.NumVariables()
    for var_name, var in vars_map.items():
        assert var_name == var.name()

    cons_map = construct_constraints_for_mvmip(
        solver=solver,
        vars_map=vars_map,
        mvmip_params=mvmip_params,
        vehicles=vehicles,
        obstacles=obstacles,
    )
    assert len(cons_map) == solver.NumConstraints()
    for cons_name, cons in cons_map.items():
        assert cons_name == cons.name()

    status = solver.Solve()

    if status == solver.OPTIMAL:
        print("Optimal solution for MVMIP found!")
        for vars_str, vars in vars_map.items():
            print(vars_str, vars.solution_value())

    else:
        print("Optimal solution for MVMIP could not be found :(")
