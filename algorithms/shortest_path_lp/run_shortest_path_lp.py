"""
Solving the shortest path problem using linear programming.
References:

1. https://www.cs.purdue.edu/homes/egrigore/580FT15/26-lp-jefferickson.pdf
2. https://people.seas.harvard.edu/~cs125/fall16/section-notes/05.pdf
"""
import cv2
from typing import List
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
) -> List[Coordinate2D]:

    assert cost_matrix[start_coord] >= 0.0, "Start in collision"
    assert cost_matrix[end_coord] >= 0.0, "End in collision"

    solver: pywraplp.Solver = pywraplp.Solver.CreateSolver("SCIP")
    objective = solver.Objective()

    m, n = cost_matrix.shape
    nodes = np.arange(m * n)
    ind_map = nodes.reshape(n, m)
    coord_map = {}

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
            coord_map[from_node] = (i, j)

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
                    solver.NumVar(0.0, 1.0, var_str)

                    # Objective
                    # Use the current node's cost as the edge weight.
                    # (Could also average current and neighbor)
                    objective.SetCoefficient(solver.LookupVariable(var_str), w)

    end_time = time.perf_counter()
    print(f"Time to setup variables and objective: {end_time - start_time} sec")

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

    node_path = []

    if status == solver.OPTIMAL:
        print("Optimal solution found!")
        # Constructing path.
        node = start_node
        node_path = [node]
        while node != end_node:
            found_shortest_path_neighbor = False
            for neighbor_node in neighbor_out_map[node]:
                var_str = _var_str(node, neighbor_node)
                assert var_str in var_strs, f"{node}, {neighbor_out_map[node]}"
                if solver.LookupVariable(var_str).solution_value() == 1.0:
                    found_shortest_path_neighbor = True
                    node = neighbor_node
                    node_path.append(neighbor_node)
                    break

            assert (
                found_shortest_path_neighbor
            ), "No neighboring edges have a solution value of 1.0. Something has gone wrong."
        assert node_path[0] == start_node
        assert node_path[-1] == end_node

    else:
        print("Optimal solution could not be found :(")

    coord_path = [coord_map[node] for node in node_path]
    for coord in coord_path:
        assert cost_matrix[coord] >= 0.0, "Invalid path found!"
    return coord_path


if __name__ == "__main__":

    n = 75
    start_coord = (0, 0)
    end_coord = (n - 1, n - 1)
    cost_matrix = np.ones((n, n), dtype=np.float64)

    obstacle_half_size = n // 8
    lower = n // 2 - obstacle_half_size
    upper = n // 2 + obstacle_half_size
    cost_matrix[lower:upper, lower:upper] = -1.0

    coord_path = find_shortest_path(
        cost_matrix=cost_matrix,
        start_coord=start_coord,
        end_coord=end_coord,
    )
    print(coord_path)

    img = np.interp(cost_matrix, [0.0, 1.0], [0, 255]).astype(np.uint8)
    for coord in coord_path:
        img[coord] = 200

    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.imshow("img", img)
    cv2.waitKey(0)
