import osqp
from scipy.sparse import csc_matrix

from common.optimization.constructs import QPInputs


def solve_qp(qp_inputs: QPInputs) -> None:
    solver = osqp.OSQP()
    # Not using **attr.asdict in case the interface changes in the future or we need to add more options.
    settings = dict(
        verbose=False,
    )
    solver.setup(
        P=csc_matrix(qp_inputs.P),
        q=qp_inputs.q,
        A=csc_matrix(qp_inputs.A),
        l=qp_inputs.lb,
        u=qp_inputs.ub,
        **settings,
    )
    return solver.solve()
