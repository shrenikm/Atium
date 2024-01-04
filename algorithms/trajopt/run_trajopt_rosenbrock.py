import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as m3
import mpl_toolkits.mplot3d.art3d as ma3
import numpy as np
from matplotlib import animation as anim
from matplotlib import cm

from algorithms.trajopt.implementation.trajopt import TrajOpt, TrajOptResult
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
    trajopt: TrajOpt,
    plot_cost: bool,
    plot_constraints: bool,
) -> None:

    assert any([plot_cost, plot_constraints]), "Something must be plotted."

    fig = plt.figure(figsize=(7, 7))
    ax: m3.Axes3D = fig.add_subplot(111, projection="3d")

    X = np.arange(-5.0, 5.1, 0.1)
    Y = np.arange(-5.0, 5.1, 0.1)
    X, Y = np.meshgrid(X, Y)
    Z_rosenbrock = rosenbrock_fn(x=X, y=Y, a=params.a, b=params.b)
    min_z, max_z = np.min(Z_rosenbrock), np.max(Z_rosenbrock)
    nzr, nzc = Z_rosenbrock.shape

    # Computing the Z's for each of the constraints so that they can be represented
    # on the plot.
    # For each X, Y, we pass the individual values into the constraint functions and
    # fill up the Z matrices.
    # This isn't really the most efficient way of doing it but is the cleanest way to
    # generically get a visual representation of each region.

    Z_lg = np.full_like(Z_rosenbrock, np.inf)
    Z_lh = np.full_like(Z_rosenbrock, np.inf)
    Z_nlg = np.full_like(Z_rosenbrock, np.inf)
    Z_nlh = np.full_like(Z_rosenbrock, np.inf)

    for i in range(nzr):
        for j in range(nzc):
            x = np.array([X[i, j], Y[i, j]], dtype=np.float64)
            # For the inequality constraints, we set Z to zero if they are satisfied.
            # As the Z's are inf by default, they won't display in the plot and doing
            # this makes sure that the constraint satisfied regions are plotted.
            # For the equality constraints, Z is zero if the constraint is zero.
            if trajopt.linear_inequality_constraints_fn is not None:
                if np.all(trajopt.linear_inequality_constraints_fn(x) <= 0):
                    Z_lg[i, j] = 0.0
            if trajopt.linear_equality_constraints_fn is not None:
                if np.allclose(trajopt.linear_equality_constraints_fn(x), 0.0):
                    Z_lh[i, j] = 0.0
            if trajopt.non_linear_inequality_constraints_fn is not None:
                if np.all(trajopt.non_linear_inequality_constraints_fn(x) <= 0):
                    Z_nlg[i, j] = 0.0
            if trajopt.non_linear_equality_constraints_fn is not None:
                if np.allclose(trajopt.non_linear_equality_constraints_fn(x), 0.0):
                    Z_nlh[i, j] = 0.0

    # Plot the cost surface.
    if plot_cost:
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

    if plot_constraints:
        ax.plot_surface(
            X, Y, Z_lg, cmap="Pastel1", linewidth=0, antialiased=False, alpha=0.5
        )
        ax.plot_surface(
            X, Y, Z_lh, cmap="Dark2", linewidth=0, antialiased=False, alpha=0.5
        )
        ax.plot_surface(
            X, Y, Z_nlg, cmap="Set3", linewidth=0, antialiased=False, alpha=0.5
        )
        ax.plot_surface(
            X, Y, Z_nlh, cmap="tab20b", linewidth=0, antialiased=False, alpha=0.5
        )

    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)
    ax.set_zlim(min_z, max_z)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    trust_region_steps = len(result)
    initial_guess_x = result[0].min_x

    # For the xyz point, we only plot the z coordinate cost if the cost surface needs to be
    # plotted. Otherwise, for just the constraint regions, we plot (x, y, 0) along the
    # ground surface as this helps in better visualization.
    xyz_point = ma3.Line3D(
        xs=[initial_guess_x[0]],
        ys=[initial_guess_x[1]],
        zs=[rosenbrock_cost_fn(initial_guess_x, params)] if plot_cost else [0.0],
        marker="o",
        markersize=7,
        color="firebrick",
    )
    xy_trajectory = ma3.Line3D(
        xs=[initial_guess_x[0]],
        ys=[initial_guess_x[1]],
        zs=[0.0],
        linestyle="dotted",
        color="rosybrown",
    )
    ax.add_line(xyz_point)
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
        z = cost if plot_cost else 0.0
        xyz_point.set_data_3d([x], [y], [z])
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

    initial_guess_x = np.array([5.0, 5.0])
    result = trajopt.solve(initial_guess_x=initial_guess_x)
    print(result.solution_x())

    _visualize_trajopt_rosenbrock_result(
        params=rosenbrock_params,
        result=result,
        trajopt=trajopt,
        plot_cost=False,
        plot_constraints=True,
    )


if __name__ == "__main__":
    run()
