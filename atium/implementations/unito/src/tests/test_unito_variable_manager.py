import numpy as np
import pytest
from pydrake.solvers import MathematicalProgram
from pydrake.symbolic import Variable

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


def test_get_c_theta_vars(
    manager: UnitoVariableManager,
    prog: MathematicalProgram,
) -> None:
    all_vars = prog.decision_variables()
    vars_c_theta = manager.get_c_theta_vars(all_vars)
    assert vars_c_theta.shape == (2 * manager.params.h * manager.params.M,)

    for i in range(len(vars_c_theta)):
        assert vars_c_theta[i].get_name() == f"{UnitoVariableManager.VARS_C_THETA_NAME}({i})"


def test_get_c_s_vars(
    manager: UnitoVariableManager,
    prog: MathematicalProgram,
) -> None:
    all_vars = prog.decision_variables()
    vars_c_s = manager.get_c_s_vars(all_vars)
    assert vars_c_s.shape == (2 * manager.params.h * manager.params.M,)

    for i in range(len(vars_c_s)):
        assert vars_c_s[i].get_name() == f"{UnitoVariableManager.VARS_C_S_NAME}({i})"


def test_get_t_vars(
    manager: UnitoVariableManager,
    prog: MathematicalProgram,
) -> None:
    all_vars = prog.decision_variables()
    vars_t = manager.get_t_vars(all_vars)
    assert vars_t.shape == (manager.params.M,)

    for i in range(len(vars_t)):
        assert vars_t[i].get_name() == f"{UnitoVariableManager.VARS_T_NAME}({i})"


def test_get_c_theta_i_vars(
    manager: UnitoVariableManager,
    prog: MathematicalProgram,
) -> None:
    all_vars = prog.decision_variables()
    for i in range(manager.params.M):
        vars_c_theta_i = manager.get_c_theta_i_vars(all_vars, i)
        assert vars_c_theta_i.shape == (2 * manager.params.h,)

        for j in range(len(vars_c_theta_i)):
            assert (
                vars_c_theta_i[j].get_name()
                == f"{UnitoVariableManager.VARS_C_THETA_NAME}({i * 2 * manager.params.h + j})"
            )


def test_get_c_s_i_vars(
    manager: UnitoVariableManager,
    prog: MathematicalProgram,
) -> None:
    all_vars = prog.decision_variables()
    for i in range(manager.params.M):
        vars_c_s_i = manager.get_c_s_i_vars(all_vars, i)
        assert vars_c_s_i.shape == (2 * manager.params.h,)

        for j in range(len(vars_c_s_i)):
            assert vars_c_s_i[j].get_name() == f"{UnitoVariableManager.VARS_C_S_NAME}({i * 2 * manager.params.h + j})"


def test_get_t_i_var(
    manager: UnitoVariableManager,
    prog: MathematicalProgram,
) -> None:
    all_vars = prog.decision_variables()
    for i in range(manager.params.M):
        var_t_i = manager.get_t_i_var(all_vars, i)
        assert isinstance(var_t_i, Variable)
        assert var_t_i.get_name() == f"{UnitoVariableManager.VARS_T_NAME}({i})"


pytest.main(["-s", "-v", __file__])
