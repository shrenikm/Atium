import osqp
from scipy.sparse import csc_matrix

from atium.core.optimization.constructs import QPInputs

OSQP_SOLVED_STATUS_STR = "solved"


def solve_qp(qp_inputs: QPInputs, verbose: bool = False) -> None:
    solver = osqp.OSQP()
    # Not using **attr.asdict in case the interface changes in the future or we need to add more options.
    settings = dict(
        verbose=verbose,
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


# TODO: Figure out this result type. OSQP documentation smh.
def is_qp_solved(osqp_results) -> bool:
    # return OSQP_SOLVED_STATUS_STR in osqp_results.info.status
    return osqp_results.info.status == OSQP_SOLVED_STATUS_STR
