import math

import numpy as np
import pytest
from pydrake.solvers import MathematicalProgram
from pydrake.symbolic import Expression, Variable

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


def test_get_t_ij_exp(
    manager: UnitoVariableManager,
    prog: MathematicalProgram,
) -> None:
    all_vars = prog.decision_variables()
    for i in range(manager.params.M):
        for j in range(manager.params.n - 1):
            t_ij_exp = manager.get_t_ij_exp(
                t_vars=manager.get_t_vars(all_vars),
                i=i,
                j=j,
            )
            assert isinstance(t_ij_exp, (float, Expression))

    # Actualy check with a regular float array.
    t_vars = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    for i in range(len(t_vars)):
        t_ij = manager.get_t_ij_exp(
            t_vars=t_vars,
            i=i,
            j=0,
        )
        np.testing.assert_allclose(t_ij, 0.0, atol=1e-6)

        t_ij = manager.get_t_ij_exp(
            t_vars=t_vars,
            i=i,
            j=manager.params.n - 1,
        )
        np.testing.assert_allclose(t_ij, t_vars[i], atol=1e-6)

    i, j = 1, 2
    t_ij = manager.get_t_ij_exp(
        t_vars=t_vars,
        i=i,
        j=j,
    )
    np.testing.assert_allclose(t_ij, t_vars[i] * j / (manager.params.n - 1), atol=1e-6)


def test_get_basis_vector_ij_exp(
    manager: UnitoVariableManager,
    prog: MathematicalProgram,
) -> None:
    all_vars = prog.decision_variables()

    t_ij_exp = manager.get_t_vars(all_vars)[0]
    for derivative in range(manager.params.h):
        basis_vector = manager.get_basis_vector_ij_exp(
            t_ij_exp=t_ij_exp,
            derivative=derivative,
        )
        assert isinstance(basis_vector, np.ndarray)
        assert basis_vector.shape == (2 * manager.params.h,)

    # Test with a regular float array.
    for derivative in range(manager.params.h):
        basis_vector = manager.get_basis_vector_ij_exp(
            t_ij_exp=0.0,
            derivative=derivative,
        )
        expected_basis_vector = np.zeros((2 * manager.params.h,), dtype=np.float64)
        expected_basis_vector[derivative] = math.factorial(derivative)
        np.testing.assert_allclose(basis_vector, expected_basis_vector, atol=1e-6)


pytest.main(["-s", "-v", __file__])
