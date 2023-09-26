import numpy as np
from ortools.linear_solver import pywraplp

solver: pywraplp.Solver = pywraplp.Solver.CreateSolver("GLOP")
print(type(solver))
assert solver is not None

x = solver.NumVar(-5., 5., "x")
y = solver.NumVar(-3., 5., "y")
solver.Add(x + 2 * y <= 3)

print(solver.NumVariables(), solver.NumConstraints())

solver.Minimize(3 * x - 7 * y)
solver.Constraint

status = solver.Solve()
if status == pywraplp.Solver.OPTIMAL:
    print("x: ", x.solution_value())
    print("y: ", y.solution_value())
