from ortools.linear_solver import pywraplp
from typing import Dict

Solver = pywraplp.Solver
SolverVariable = pywraplp.Variable
SolverVariableMap = Dict[str, SolverVariable]
SolverConstraint = pywraplp.Constraint
SolverConstraintMap = Dict[str, pywraplp.Constraint]
