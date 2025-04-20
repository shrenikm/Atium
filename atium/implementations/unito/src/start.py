from pydrake.solvers import MathematicalProgram, Solve

prog = MathematicalProgram()


x = prog.NewContinuousVariables(1, "x")
y = prog.NewContinuousVariables(1, "y")
c = prog.AddConstraint(x[0] + y[0] == 1)
prog.AddCost(x[0] ** 2 + y[0] ** 2)
print(prog)


def f(x):
    print(prog.EvalBinding(c, [x[0], x[1]]))


def cc(vars):
    x, y = vars
    if x > 0 and y > 0:
        print("here!")
        return x + y
    print("not here!")
    return x + y - 1


prog.AddVisualizationCallback(f, [x, y])
prog.AddCost(cc, [x, y])


res = Solve(prog, [0, 0])
print(res.is_success())
print(res.GetSolution(x))
print(res.GetSolution(y))
print(res.get_solver_id().name())
print(res.get_solution_result())
