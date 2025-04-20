import numpy as np
import pytest
from pydrake.solvers import MathematicalProgram

from atium.implementations.unito.src.unito_utils import UnitoParams
from atium.implementations.unito.src.unito_variable_manager import UnitoVariableManager


@pytest.fixture(scope="module")
def params() -> UnitoParams:
    return UnitoParams(
        h=3,
        M=3,
        n=4,
        epsilon_t=0.1,
        W=np.ones((2, 2), dtype=np.float64) * 0.1,
    )


@pytest.fixture(scope="module")
def manager(params: UnitoParams) -> UnitoVariableManager:
    return UnitoVariableManager(params=params)


@pytest.fixture(scope="function")
def prog(manager: UnitoVariableManager) -> MathematicalProgram:
    prog = MathematicalProgram()
    manager.create_decision_variables(prog)
    return prog


def test_create_decision_variables(
    manager: UnitoVariableManager,
    prog: MathematicalProgram,
) -> None:
    assert prog.num_vars() == 4 * manager.params.h * manager.params.M + manager.params.M


pytest.main(["-s", "-v", __file__])
