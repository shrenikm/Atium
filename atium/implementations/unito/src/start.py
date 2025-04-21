import numpy as np
from pydrake.solvers import MathematicalProgram, Solve

prog = MathematicalProgram()


x = prog.NewContinuousVariables(1, "x")
y = prog.NewContinuousVariables(1, "y")
c = prog.AddConstraint(x[0] + y[0] == 1)
prog.AddCost(x[0] ** 2 + y[0] ** 2)


def f(x):
    print(prog.EvalBinding(c, [x[0], x[1]]))


def cc(vars):
    x, y = vars
    cost = 0.
    cost += x**2
    print("type", type(cost))
    return x**2


prog.AddVisualizationCallback(f, [x, y])
prog.AddCost(cc, np.array([x, y]))

print(prog)

res = Solve(prog, [0, 0])
print(res.is_success())
print(res.GetSolution(x))
print(res.GetSolution(y))
print(res.get_solver_id().name())
print(res.get_solution_result())
