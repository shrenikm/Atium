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

    cons_map = construct_constraints_for_mvmip(
        solver=solver,
        vars_map=vars_map,
        mvmip_params=mvmip_params,
        vehicles=vehicles,
        obstacles=obstacles,
    )
    assert len(cons_map) == solver.NumConstraints()
