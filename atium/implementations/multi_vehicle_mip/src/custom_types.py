from typing import Any, Dict

from ortools.linear_solver import pywraplp

from atium.core.utils.custom_types import ControlTrajectoryArray, StateTrajectoryArray

Solver = pywraplp.Solver
SolverVariable = pywraplp.Variable
SolverVariableMap = Dict[str, SolverVariable]
SolverConstraint = pywraplp.Constraint
SolverConstraintMap = Dict[str, pywraplp.Constraint]
SolverObjective = pywraplp.Objective
VehicleStateTrajectoryMap = Dict[int, StateTrajectoryArray]
VehicleControlTrajectoryMap = Dict[int, ControlTrajectoryArray]

SetupYamlDict = Dict[str, Any]
