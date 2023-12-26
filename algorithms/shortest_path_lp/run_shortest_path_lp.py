"""
Solving the shortest path problem using linear programming.
References:

1. https://www.cs.purdue.edu/homes/egrigore/580FT15/26-lp-jefferickson.pdf
2. https://people.seas.harvard.edu/~cs125/fall16/section-notes/05.pdf
"""
import numpy as np
import time

from common.custom_types import Coordinate2D, CostMatrix

from ortools.linear_solver import pywraplp

NEIGHBOR_DELTA_I = [0, -1, -1, -1, 0, 1, 1, 1]
NEIGHBOR_DELTA_J = [1, 1, 0, -1, -1, -1, 0, 1]


def _var_str(
    from_node: int,
    to_node: int,
) -> str:
    return f"x_{from_node}_{to_node}"


def _start_cons_var_str() -> str:
    return f"c_s"


def _end_cons_var_str() -> str:
    return f"c_e"


def _graph_cons_var_str(node: int) -> str:
    return f"c_g_{node}"


def find_shortest_path(
    cost_matrix: CostMatrix,
    start_coord: Coordinate2D,
    end_coord: Coordinate2D,
) -> None:

    assert cost_matrix[start_coord] >= 0.0, "Start in collision"
    assert cost_matrix[end_coord] >= 0.0, "End in collision"

    solver: pywraplp.Solver = pywraplp.Solver.CreateSolver("SCIP")
    objective = solver.Objective()

    m, n = cost_matrix.shape
    nodes = np.arange(m * n)
    ind_map = nodes.reshape(n, m)

    start_node = ind_map[start_coord]
    end_node = ind_map[end_coord]

    neighbor_out_map = {node: [] for node in nodes}
    neighbor_in_map = {node: [] for node in nodes}
    var_strs = []
    cons_strs = []

    # Set up variables.
    start_time = time.perf_counter()
    for i in range(m):
        for j in range(n):

            from_coord = (i, j)
            from_node = ind_map[from_coord]

            if cost_matrix[from_coord] < 0.0:
                continue

            w = cost_matrix[from_coord]

            for delta_i, delta_j in zip(NEIGHBOR_DELTA_I, NEIGHBOR_DELTA_J):
                neighbor_coord = (i + delta_i, j + delta_j)
                if (
                    neighbor_coord[0] >= 0
                    and neighbor_coord[0] < m
                    and neighbor_coord[1] >= 0
                    and neighbor_coord[1] < n
                ):

                    if cost_matrix[neighbor_coord] < 0.0:
                        continue

                    to_node = ind_map[neighbor_coord]

                    # Add to neighbor maps and create variable.
                    neighbor_out_map[from_node].append(to_node)
                    neighbor_in_map[to_node].append(from_node)

                    # Variable str + uniqueness check
                    var_str = _var_str(from_node=from_node, to_node=to_node)
                    assert var_str not in var_strs
                    var_strs.append(var_str)
                    #solver.NumVar(0.0, 1.0, var_str)
                    solver.NumVar(-solver.infinity(), solver.infinity(), var_str)

                    # Objective
                    # Use the current node's cost as the edge weight.
                    # (Could also average current and neighbor)
                    objective.SetCoefficient(solver.LookupVariable(var_str), w)

    end_time = time.perf_counter()
    print(f"Time to setup variables and objective: {end_time - start_time} sec")

    print(neighbor_out_map)
    print("===")
    print(neighbor_in_map)
    print("===")
    print(ind_map)

    # Constraints
    start_time = time.perf_counter()
    cons_str = _start_cons_var_str()
    assert cons_str not in cons_strs
    start_constraint = solver.Constraint(1.0, 1.0, cons_str)
    for neighbor_node in neighbor_out_map[start_node]:
        var_str = _var_str(start_node, neighbor_node)
        start_constraint.SetCoefficient(solver.LookupVariable(var_str), 1.0)

    for neighbor_node in neighbor_in_map[start_node]:
        var_str = _var_str(neighbor_node, start_node)
        start_constraint.SetCoefficient(solver.LookupVariable(var_str), -1.0)

    cons_str = _end_cons_var_str()
    assert cons_str not in cons_strs
    end_constraint = solver.Constraint(-1.0, -1.0, cons_str)
    for neighbor_node in neighbor_out_map[end_node]:
        var_str = _var_str(end_node, neighbor_node)
        end_constraint.SetCoefficient(solver.LookupVariable(var_str), 1.0)

    for neighbor_node in neighbor_in_map[end_node]:
        var_str = _var_str(neighbor_node, end_node)
        end_constraint.SetCoefficient(solver.LookupVariable(var_str), -1.0)

    for node in nodes:
        if node == start_node or node == end_node:
            continue
        out_nodes = neighbor_out_map[node]
        in_nodes = neighbor_in_map[node]

        if len(out_nodes) or len(in_nodes):
            cons_str = _graph_cons_var_str(node=node)
            assert cons_str not in cons_strs
            graph_constraint = solver.Constraint(0.0, 0.0, cons_str)
            for out_node in out_nodes:
                var_str = _var_str(node, out_node)
                graph_constraint.SetCoefficient(solver.LookupVariable(var_str), 1.0)
            for in_node in in_nodes:
                var_str = _var_str(in_node, node)
                graph_constraint.SetCoefficient(solver.LookupVariable(var_str), -1.0)

    end_time = time.perf_counter()
    print(f"Time to setup constraints: {end_time - start_time} sec")

    objective.SetMinimization()

    start_time = time.perf_counter()
    status = solver.Solve()
    end_time = time.perf_counter()
    print(f"Time to solve: {end_time - start_time} sec")

    for var_str in var_strs:
        print(var_str, solver.LookupVariable(var_str).solution_value())

    if status == solver.OPTIMAL:
        print("Optimal solution found!")
    else:
        print("Optimal solution could not be found :(")


if __name__ == "__main__":

    n = 10
    start_coord = (0, 0)
    end_coord = (n - 1, n - 1)
    cost_matrix = np.ones((n, n), dtype=np.float64)
    cost_matrix[3, 2] = -1.0

    find_shortest_path(
        cost_matrix=cost_matrix,
        start_coord=start_coord,
        end_coord=end_coord,
    )
