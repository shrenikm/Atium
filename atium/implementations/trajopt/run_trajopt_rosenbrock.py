import attr
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as m3
import mpl_toolkits.mplot3d.art3d as ma3
import numpy as np
from matplotlib import animation as anim
from matplotlib import cm

from atium.core.optimization.derivative_splicer import DerivativeSplicedConstraintsFn, DerivativeSplicedCostFn
from atium.core.optimization.standard_functions.rosenbrock import RosenbrockParams, rosenbrock_cost_fn, rosenbrock_fn
from atium.core.utils.custom_types import VectorNf64
from atium.core.utils.file_utils import get_file_path_in_implementations_results_dir
from atium.implementations.trajopt.src.trajopt import TrajOpt, TrajOptParams, TrajOptResult


def _visualize_trajopt_rosenbrock_result(
    params: RosenbrockParams,
    result: TrajOptResult,
    trajopt: TrajOpt,
    plot_cost: bool,
    plot_constraints: bool,
    setup_num: int,
    save_video: bool,
) -> None:
    assert any([plot_cost, plot_constraints]), "Something must be plotted."

    fig = plt.figure(figsize=(7, 7))
    ax: m3.Axes3D = fig.add_subplot(111, projection="3d")

    # More of a 3D view if the cost surface needs to be plotted.
    # For just the constraints, we go for a more flat view of the x-y grid.
    if plot_cost:
        ax.view_init(elev=20.0, azim=110.0, roll=0.0)
    else:
        # For the flat view, we don't want to be plotting the numbers on the z axis
        # as they would overlap and look bad.
        ax.set_zticklabels([])
        ax.view_init(elev=90.0, azim=90.0, roll=0.0)

    resolution = 0.05
    X = np.arange(-5.0, 5.1, resolution)
    Y = np.arange(-5.0, 5.1, resolution)
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
                # Requires a generous atol here that is > resolution, otherwise nothing would plot.
                if np.allclose(trajopt.linear_equality_constraints_fn(x), 0.0, atol=2 * resolution):
                    Z_lh[i, j] = 0.0
            if trajopt.non_linear_inequality_constraints_fn is not None:
                if np.all(trajopt.non_linear_inequality_constraints_fn(x) <= 0):
                    Z_nlg[i, j] = 0.0
            if trajopt.non_linear_equality_constraints_fn is not None:
                # Requires a generous atol here that is > resolution, otherwise nothing would plot.
                if np.allclose(
                    trajopt.non_linear_equality_constraints_fn(x),
                    0.0,
                    atol=2 * resolution,
                ):
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
        ax.plot_surface(X, Y, Z_lg, cmap="Pastel1", linewidth=0, antialiased=False, alpha=0.5)
        ax.plot_surface(X, Y, Z_lh, cmap="Dark2", linewidth=0, antialiased=False, alpha=0.5)
        ax.plot_surface(X, Y, Z_nlg, cmap="Set3", linewidth=0, antialiased=False, alpha=0.5)
        ax.plot_surface(X, Y, Z_nlh, cmap="tab20b", linewidth=0, antialiased=False, alpha=0.5)

    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.5, 5.5)
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
        x, y = entry.updated_min_x if entry.improvement else entry.min_x
        cost = entry.cost
        x_trajectory = [entry.min_x[0] for entry in result.entries[: trust_region_iter + 1]]
        y_trajectory = [entry.min_x[1] for entry in result.entries[: trust_region_iter + 1]]
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
        interval=300,
        repeat=True,
    )

    if save_video:
        postfix_list = []
        if plot_cost:
            postfix_list.append("cost")
        if plot_constraints:
            postfix_list.append("constraint")
        output_filename = f"trajopt_rosenbrock_{setup_num}_{'_'.join(postfix_list)}.gif"
        output_video_path = get_file_path_in_implementations_results_dir(
            output_filename=output_filename,
        )
        animation.save(
            filename=output_video_path,
        )
        print(f"Animation saved to {output_video_path}")

    plt.show()


@attr.frozen
class RosenbrockOptParamsConstructor:
    params: RosenbrockParams

    def __call__(self, x: VectorNf64) -> RosenbrockParams:
        del x
        return self.params


def lg_fn1(z: VectorNf64) -> VectorNf64:
    """
    Linear inequality constraints of the form:
    x >= c, y >= d
    (x, y) >= (c, d) => (x-c, y-d) >= 0 => (c-x, d-y) <= 0
    or x <=c, y <= d
    (x, y) <= (c, d) => (x-c, y=d) <= 0
    """
    x, y = z
    # x <= -2., y >= 0.
    return jnp.array([x + 2.0, 0.0 - y], dtype=jnp.float32)


def lg_fn2(z: VectorNf64) -> VectorNf64:
    """
    Linear inequality constraints of the form:
    x >= c, y >= d
    (x, y) >= (c, d) => (x-c, y-d) >= 0 => (c-x, d-y) <= 0
    or x <=c, y <= d
    (x, y) <= (c, d) => (x-c, y=d) <= 0
    """
    x, y = z
    # x >= 2., y >= -5.
    return jnp.array([2.0 - x, -5.0 - y], dtype=jnp.float32)


def nlg_fn1(z: VectorNf64) -> VectorNf64:
    """
    Non linear inequality (circle) constraints of the form:
    (x - xc)^2 + (y - yc)^2 - r^2 <= 0
    """
    x, y = z
    xc, yc = (2.0, 2.0)
    r = 2.0
    return jnp.array((x - xc) ** 2 + (y - yc) ** 2 - r**2, dtype=jnp.float32)


def nlg_fn2(z: VectorNf64) -> VectorNf64:
    """
    Two circle constraints.
    One at (2., 2.) with a radius of 2.
    One at (4., 0.) with a radius of 1.5
    """
    x, y = z
    xc1, yc1 = (2.0, 2.0)
    r1 = 2.0
    xc2, yc2 = (4.0, 1.0)
    r2 = 2.5
    return jnp.array(
        [
            (x - xc1) ** 2 + (y - yc1) ** 2 - r1**2,
            (x - xc2) ** 2 + (y - yc2) ** 2 - r2**2,
        ],
        dtype=jnp.float32,
    )


def nlh_fn1(z: VectorNf64) -> VectorNf64:
    """
    Non linear equality (circle) constraints of the form:
    (x - xc)^2 + (y - yc)^2 - r^2 == 0
    """
    x, y = z
    xc, yc = (2.0, 2.0)
    r = 1.0
    return jnp.array((x - xc) ** 2 + (y - yc) ** 2 - r**2, dtype=jnp.float32)


def run_trajopt(setup_num: int) -> None:
    rosenbrock_params = RosenbrockParams(a=1.0, b=100.0)
    trajopt_params = TrajOptParams(
        mu_0=1.0,
        s_0=1e-4,
        c=1e-2,
        k=10.0,
        f_tol=1e-4,
        x_tol=1e-4,
        c_tol=1e-2,
        tau_plus=1.5,
        tau_minus=0.1,
        tau_max=10.0,
        tau_min=1e-4,
        max_iter=200,
        second_order_inequalities=True,
        second_order_equalities=True,
    )

    rosenbrock_params_constructor = RosenbrockOptParamsConstructor(params=rosenbrock_params)
    cost_fn_ds = DerivativeSplicedCostFn(
        core_fn=rosenbrock_cost_fn,
        use_jit=True,
        construct_params_fn=rosenbrock_params_constructor,
    )

    lg_fn_ds1 = DerivativeSplicedConstraintsFn(
        core_fn=lg_fn1,
        use_jit=True,
    )
    lg_fn_ds2 = DerivativeSplicedConstraintsFn(
        core_fn=lg_fn2,
        use_jit=True,
    )
    nlg_fn_ds1 = DerivativeSplicedConstraintsFn(
        core_fn=nlg_fn1,
        use_jit=True,
    )
    nlg_fn_ds2 = DerivativeSplicedConstraintsFn(
        core_fn=nlg_fn2,
        use_jit=True,
    )
    nlh_fn_ds1 = DerivativeSplicedConstraintsFn(
        core_fn=nlh_fn1,
        use_jit=True,
    )

    if setup_num == 1:
        initial_guess_x = np.array([-1.0, -2.0])
        trajopt = TrajOpt(
            params=trajopt_params,
            cost_fn=cost_fn_ds,
        )

    elif setup_num == 2:
        initial_guess_x = np.array([5.0, 5.0])
        trajopt = TrajOpt(
            params=trajopt_params,
            cost_fn=cost_fn_ds,
        )
    elif setup_num == 3:
        initial_guess_x = np.array([-5.0, 5.0])
        trajopt = TrajOpt(
            params=trajopt_params,
            cost_fn=cost_fn_ds,
            linear_inequality_constraints_fn=lg_fn_ds1,
        )
    elif setup_num == 4:
        initial_guess_x = np.array([5.0, -5.0])
        trajopt = TrajOpt(
            params=trajopt_params,
            cost_fn=cost_fn_ds,
            linear_inequality_constraints_fn=lg_fn_ds2,
            non_linear_inequality_constraints_fn=nlg_fn_ds1,
        )
    elif setup_num == 5:
        initial_guess_x = np.array([5.0, -5.0])
        trajopt = TrajOpt(
            params=trajopt_params,
            cost_fn=cost_fn_ds,
            linear_inequality_constraints_fn=lg_fn_ds2,
            non_linear_inequality_constraints_fn=nlg_fn_ds1,
            non_linear_equality_constraints_fn=nlh_fn_ds1,
        )

    elif setup_num == 6:
        initial_guess_x = np.array([5.0, -5.0])
        trajopt = TrajOpt(
            params=trajopt_params,
            cost_fn=cost_fn_ds,
            linear_inequality_constraints_fn=lg_fn_ds2,
            non_linear_inequality_constraints_fn=nlg_fn_ds2,
            non_linear_equality_constraints_fn=nlh_fn_ds1,
        )
    else:
        raise NotImplementedError(f"Invalid setup number: {setup_num}")

    result = trajopt.solve(initial_guess_x=initial_guess_x)

    _visualize_trajopt_rosenbrock_result(
        params=rosenbrock_params,
        result=result,
        trajopt=trajopt,
        plot_cost=True,
        plot_constraints=True,
        setup_num=setup_num,
        save_video=False,
    )


if __name__ == "__main__":
    setup_num = 6
    run_trajopt(setup_num=setup_num)
