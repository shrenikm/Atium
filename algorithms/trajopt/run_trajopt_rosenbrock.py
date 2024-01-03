import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as ma3
import numpy as np
from matplotlib import animation as anim
from matplotlib import cm

from algorithms.trajopt.implementation.trajopt import TrajOptResult
from algorithms.trajopt.setups.trajopt_rosenbrock_setup import (
    setup_trajopt_for_rosenbrock,
)
from common.custom_types import VectorNf64
from common.optimization.standard_functions.rosenbrock import (
    RosenbrockParams,
    rosenbrock_cost_fn,
    rosenbrock_fn,
)


# TODO: Move elsewhere.
def _visualize_trajopt_rosenbrock_result(
    params: RosenbrockParams,
    result: TrajOptResult,
) -> None:

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")

    X = np.arange(-5.0, 5.1, 0.1)
    Y = np.arange(-5.0, 5.1, 0.1)
    X, Y = np.meshgrid(X, Y)
    Z_rosenbrock = rosenbrock_fn(x=X, y=Y, a=params.a, b=params.b)
    min_z, max_z = np.min(Z_rosenbrock), np.max(Z_rosenbrock)
    ax.plot_surface(
        X,
        Y,
        Z_rosenbrock,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
        alpha=0.4,
        edgecolor="none",
    )

    def _constraint_plane1(x, y):
        Z = np.zeros_like(x)
        Z[np.where(np.logical_and(x >= 2.0, y >= 2.0))] = np.inf
        return Z

    Z_constraint1 = _constraint_plane1(X, Y)

    ax.plot_surface(
        X, Y, Z_constraint1, cmap="Dark2", linewidth=0, antialiased=False, alpha=0.5
    )

    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)
    ax.set_zlim(min_z, max_z)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    trust_region_steps = len(result)
    initial_guess_x = result[0].min_x

    xy_point = ma3.Line3D(
        xs=[initial_guess_x[0]],
        ys=[initial_guess_x[1]],
        zs=[0.0],
        marker="o",
        markersize=5,
        color="indianred",
    )
    xyz_point = ma3.Line3D(
        xs=[initial_guess_x[0]],
        ys=[initial_guess_x[1]],
        zs=[rosenbrock_cost_fn(initial_guess_x, params)],
        marker="x",
        markersize=1,
        color="black",
    )
    xyz_projection_line = ma3.Line3D(
        xs=[initial_guess_x[0], initial_guess_x[0]],
        ys=[initial_guess_x[1], initial_guess_x[1]],
        zs=[0.0, rosenbrock_cost_fn(initial_guess_x, params)],
        linestyle="dotted",
        color="gray",
    )
    xy_trajectory = ma3.Line3D(
        xs=[initial_guess_x[0]],
        ys=[initial_guess_x[1]],
        zs=[0.0],
        linestyle="dotted",
        color="rosybrown",
    )
    ax.add_line(xy_point)
    ax.add_line(xyz_point)
    ax.add_line(xyz_projection_line)
    ax.add_line(xy_trajectory)

    def anim_update(trust_region_iter):
        entry = result[trust_region_iter]
        x, y = entry.min_x
        cost = entry.cost
        x_trajectory = [
            entry.min_x[0] for entry in result.entries[: trust_region_iter + 1]
        ]
        y_trajectory = [
            entry.min_x[1] for entry in result.entries[: trust_region_iter + 1]
        ]
        ax.set_title(
            f"""
            Rosenbrock TrajOpt trust region step: {trust_region_iter + 1}/{trust_region_steps}
            x: [{x:.3f}, {y:.3f}]
            """
        )
        # print(trust_region_iter, f"[{x}, {y}]", cost)
        # print(cost, rosenbrock_cost_fn([x, y], params))
        # print("============")
        xy_point.set_data_3d([x], [y], [0.0])
        xyz_point.set_data_3d([x], [y], [cost])
        xyz_projection_line.set_data_3d([x, x], [y, y], [0.0, cost])
        xy_trajectory.set_data_3d(x_trajectory, y_trajectory, [0] * len(x_trajectory))

    animation = anim.FuncAnimation(
        fig=fig,
        func=anim_update,
        frames=trust_region_steps,
        interval=200,
        repeat=True,
    )

    plt.show()


def run() -> None:
    rosenbrock_params = RosenbrockParams(a=1.0, b=100.0)
    trajopt = setup_trajopt_for_rosenbrock(
        rosenbrock_params=rosenbrock_params,
    )

    # _visualze(params=rosenbrock_params)

    initial_guess_x = np.array([4., 2.])
    result = trajopt.solve(initial_guess_x=initial_guess_x)
    print(result.solution_x())

    _visualize_trajopt_rosenbrock_result(
        params=rosenbrock_params,
        result=result,
    )


if __name__ == "__main__":
    run()
