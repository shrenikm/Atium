import attr

from atium.core.utils.custom_types import VectorNf64


@attr.frozen
class RosenbrockParams:
    a: float
    b: float


def rosenbrock_cost_fn(z: VectorNf64, params: RosenbrockParams) -> float:
    x, y = z
    return (params.a - x) ** 2 + params.b * (y - x**2) ** 2


def rosenbrock_fn(x: float, y: float, a: float, b: float) -> float:
    return (a - x) ** 2 + b * (y - x**2) ** 2
