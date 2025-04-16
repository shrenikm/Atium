import attr

from atium.core.utils.custom_types import MatrixMNf64, MatrixNNf64, VectorNf64


@attr.frozen
class QPInputs:
    """
    Inputs to a QP optimization problem
    J = 0.5 * x^T @ P @ x + q^T @ x
    s.t lb <= Ax <= ub
    """

    P: MatrixNNf64
    q: VectorNf64
    A: MatrixMNf64
    lb: VectorNf64
    ub: VectorNf64
