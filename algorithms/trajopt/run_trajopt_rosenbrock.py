import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from algorithms.trajopt.setups.trajopt_rosenbrock_setup import (
    setup_trajopt_for_rosenbrock,
)
from common.optimization.standard_functions.rosenbrock import (
    RosenbrockParams,
    rosenbrock_fn,
)


# TODO: Move elsewhere.
def _visualze(
    params: RosenbrockParams,
) -> None:

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")

    X = np.arange(-2.0, 2.1, 0.1)
    Y = np.arange(-2.0, 2.1, 0.1)
    X, Y = np.meshgrid(X, Y)
    Z = rosenbrock_fn(x=X, y=Y, a=params.a, b=params.b)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    plt.show()


def run() -> None:
    rosenbrock_params = RosenbrockParams(a=1.0, b=100.0)
    trajopt = setup_trajopt_for_rosenbrock(
        rosenbrock_params=rosenbrock_params,
    )

    # _visualze(params=rosenbrock_params)

    initial_guess_x = np.array([100.0, 100.0])
    result = trajopt.solve(initial_guess_x=initial_guess_x)


if __name__ == "__main__":
    run()
