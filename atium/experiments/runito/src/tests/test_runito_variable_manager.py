import math

import numpy as np
import pytest
from pydrake.solvers import MathematicalProgram
from pydrake.symbolic import Expression, Variable

from atium.core.utils.custom_types import DecisionVariablesVector
from atium.experiments.runito.src.runito_utils import RunitoParams
from atium.experiments.runito.src.runito_variable_manager import RunitoVariableManager


@pytest.fixture(scope="module")
def rng() -> np.random.RandomState:
    return np.random.RandomState(7)


@pytest.fixture(scope="module")
def params() -> RunitoParams:
    return RunitoParams(
        h=3,
        M=3,
        n=4,
        epsilon_t=0.1,
        W=np.ones((2, 2), dtype=np.float64) * 0.1,
    )


@pytest.fixture(scope="module")
def all_vars_values(params: RunitoParams) -> DecisionVariablesVector:
    """
    Floating point decision variables for testing.
    c_x_i_j will be a number 1{i}{j} where i is the segment and j is the sample.
    c_y_i_j will be a number 2{i}{j} where i is the segment and j is the sample.
    c_theta_i_j will be a number 3{i}{j} where i is the segment and j is the sample.
    theta_i  will be a number 4{i} where i is the segment.
    """
    c_x = np.array([int(f"1{i}{j}") for i in range(params.M) for j in range(2 * params.h)], dtype=np.float64)
    c_y = np.array([int(f"2{i}{j}") for i in range(params.M) for j in range(2 * params.h)], dtype=np.float64)
    c_theta = np.array([int(f"3{i}{j}") for i in range(params.M) for j in range(2 * params.h)], dtype=np.float64)
    t = np.array([int(f"4{i}") for i in range(params.M)], dtype=np.float64)
    return np.concatenate((c_x, c_y, c_theta, t), axis=0)


@pytest.fixture(scope="module")
def manager(params: RunitoParams) -> RunitoVariableManager:
    return RunitoVariableManager(params=params)


@pytest.fixture(scope="function")
def prog(manager: RunitoVariableManager) -> MathematicalProgram:
    prog = MathematicalProgram()
    manager.create_decision_variables(prog)
    return prog


def test_create_decision_variables(
    manager: RunitoVariableManager,
    prog: MathematicalProgram,
) -> None:
    assert prog.num_vars() == 6 * manager.params.h * manager.params.M + manager.params.M


def test_get_c_x_vars(
    manager: RunitoVariableManager,
    prog: MathematicalProgram,
    all_vars_values: DecisionVariablesVector,
) -> None:
    all_vars = prog.decision_variables()
    vars_c_x = manager.get_c_x_vars(all_vars)
    assert vars_c_x.shape == (2 * manager.params.h * manager.params.M,)

    for i in range(len(vars_c_x)):
        assert vars_c_x[i].get_name() == f"{RunitoVariableManager.VARS_C_X_NAME}({i})"

    # Test with actual values.
    vars_c_x_values = manager.get_c_x_vars(all_vars_values)
    for value in vars_c_x_values:
        assert str(value)[0] == "1"


def test_get_c_y_vars(
    manager: RunitoVariableManager,
    prog: MathematicalProgram,
    all_vars_values: DecisionVariablesVector,
) -> None:
    all_vars = prog.decision_variables()
    vars_c_y = manager.get_c_y_vars(all_vars)
    assert vars_c_y.shape == (2 * manager.params.h * manager.params.M,)

    for i in range(len(vars_c_y)):
        assert vars_c_y[i].get_name() == f"{RunitoVariableManager.VARS_C_Y_NAME}({i})"

    # Test with actual values.
    vars_c_y_values = manager.get_c_y_vars(all_vars_values)
    for value in vars_c_y_values:
        assert str(value)[0] == "2"


def test_get_c_theta_vars(
    manager: RunitoVariableManager,
    prog: MathematicalProgram,
    all_vars_values: DecisionVariablesVector,
) -> None:
    all_vars = prog.decision_variables()
    vars_c_theta = manager.get_c_theta_vars(all_vars)
    assert vars_c_theta.shape == (2 * manager.params.h * manager.params.M,)

    for i in range(len(vars_c_theta)):
        assert vars_c_theta[i].get_name() == f"{RunitoVariableManager.VARS_C_THETA_NAME}({i})"

    # Test with actual values.
    vars_c_theta_values = manager.get_c_theta_vars(all_vars_values)
    for value in vars_c_theta_values:
        assert str(value)[0] == "3"


def test_get_t_vars(
    manager: RunitoVariableManager,
    prog: MathematicalProgram,
    all_vars_values: DecisionVariablesVector,
) -> None:
    all_vars = prog.decision_variables()
    vars_t = manager.get_t_vars(all_vars)
    assert vars_t.shape == (manager.params.M,)

    for i in range(len(vars_t)):
        assert vars_t[i].get_name() == f"{RunitoVariableManager.VARS_T_NAME}({i})"

    # Test with actual values.
    vars_t_values = manager.get_t_vars(all_vars_values)
    for value in vars_t_values:
        assert str(value)[0] == "4"


def test_get_c_x_i_vars(
    manager: RunitoVariableManager,
    prog: MathematicalProgram,
    all_vars_values: DecisionVariablesVector,
) -> None:
    all_vars = prog.decision_variables()
    for i in range(manager.params.M):
        vars_c_x_i = manager.get_c_x_i_vars(all_vars, i)
        assert vars_c_x_i.shape == (2 * manager.params.h,)

        for j in range(len(vars_c_x_i)):
            assert vars_c_x_i[j].get_name() == f"{RunitoVariableManager.VARS_C_X_NAME}({i * 2 * manager.params.h + j})"

    # Test with actual values.
    for i in range(manager.params.M):
        # Test with actual values.
        vars_c_x_i_values = manager.get_c_x_i_vars(all_vars_values, i)
        for j in range(manager.params.n):
            value = vars_c_x_i_values[j]
            assert str(value)[0] == "1"
            assert str(value)[1:].startswith(f"{i}{j}")


def test_get_c_y_i_vars(
    manager: RunitoVariableManager,
    prog: MathematicalProgram,
    all_vars_values: DecisionVariablesVector,
) -> None:
    all_vars = prog.decision_variables()
    for i in range(manager.params.M):
        vars_c_y_i = manager.get_c_y_i_vars(all_vars, i)
        assert vars_c_y_i.shape == (2 * manager.params.h,)

        for j in range(len(vars_c_y_i)):
            assert vars_c_y_i[j].get_name() == f"{RunitoVariableManager.VARS_C_Y_NAME}({i * 2 * manager.params.h + j})"

    # Test with actual values.
    for i in range(manager.params.M):
        # Test with actual values.
        vars_c_y_i_values = manager.get_c_y_i_vars(all_vars_values, i)
        for j in range(manager.params.n):
            value = vars_c_y_i_values[j]
            assert str(value)[0] == "2"
            assert str(value)[1:].startswith(f"{i}{j}")


def test_get_c_theta_i_vars(
    manager: RunitoVariableManager,
    prog: MathematicalProgram,
    all_vars_values: DecisionVariablesVector,
) -> None:
    all_vars = prog.decision_variables()
    for i in range(manager.params.M):
        vars_c_theta_i = manager.get_c_theta_i_vars(all_vars, i)
        assert vars_c_theta_i.shape == (2 * manager.params.h,)

        for j in range(len(vars_c_theta_i)):
            assert (
                vars_c_theta_i[j].get_name()
                == f"{RunitoVariableManager.VARS_C_THETA_NAME}({i * 2 * manager.params.h + j})"
            )

    # Test with actual values.
    for i in range(manager.params.M):
        # Test with actual values.
        vars_c_theta_i_values = manager.get_c_theta_i_vars(all_vars_values, i)
        for j in range(manager.params.n):
            value = vars_c_theta_i_values[j]
            assert str(value)[0] == "3"
            assert str(value)[1:].startswith(f"{i}{j}")


def test_get_t_i_var(
    manager: RunitoVariableManager,
    prog: MathematicalProgram,
    all_vars_values: DecisionVariablesVector,
) -> None:
    all_vars = prog.decision_variables()
    for i in range(manager.params.M):
        var_t_i = manager.get_t_i_var(all_vars, i)
        assert isinstance(var_t_i, Variable)
        assert var_t_i.get_name() == f"{RunitoVariableManager.VARS_T_NAME}({i})"

    # Test with actual values.
    for i in range(manager.params.M):
        # Test with actual values.
        value = manager.get_t_i_var(all_vars_values, i)
        assert str(value)[0] == "4"
        assert str(value)[1:].startswith(f"{i}")


def test_compute_t_ijl_exp(
    manager: RunitoVariableManager,
    prog: MathematicalProgram,
    all_vars_values: DecisionVariablesVector,
) -> None:
    all_vars = prog.decision_variables()
    for i in range(manager.params.M):
        t_i_var = manager.get_t_i_var(all_vars, i)
        for j in range(manager.params.n):
            for l in [0, 1, 2]:  # noqa: E741
                t_ijl_exp = manager.compute_t_ijl_exp(
                    t_i_var=t_i_var,
                    j=j,
                    l=l,
                )
                assert isinstance(t_ijl_exp, Expression)

    # Test with actual values.
    for i in range(manager.params.M):
        t_i_value = manager.get_t_i_var(all_vars_values, i)
        for j in range(manager.params.n):
            for l in [0, 1, 2]:  # noqa: E741
                t_ijl_value = manager.compute_t_ijl_exp(
                    t_i_var=t_i_value,
                    j=j,
                    l=l,
                )
                expected_t_ijl_value = (j + 0.5 * l) * t_i_value / manager.params.n
                np.testing.assert_allclose(t_ijl_value, expected_t_ijl_value, atol=1e-12)


def test_compute_basis_vector_exp(
    manager: RunitoVariableManager,
    prog: MathematicalProgram,
) -> None:
    all_vars = prog.decision_variables()

    t_ij_exp = manager.get_t_vars(all_vars)[0]
    for derivative in range(manager.params.h):
        basis_vector = manager.compute_basis_vector_exp(
            t_exp=t_ij_exp,
            derivative=derivative,
        )
        assert isinstance(basis_vector, np.ndarray)
        assert basis_vector.shape == (2 * manager.params.h,)

    # Test with actual values.
    for derivative in range(manager.params.h):
        basis_vector = manager.compute_basis_vector_exp(
            t_exp=0.0,
            derivative=derivative,
        )
        expected_basis_vector = np.zeros((2 * manager.params.h,), dtype=np.float64)
        expected_basis_vector[derivative] = math.factorial(derivative)
        np.testing.assert_allclose(basis_vector, expected_basis_vector, atol=1e-12)


def test_compute_sigma_i_exp(
    rng: np.random.RandomState,
    manager: RunitoVariableManager,
    prog: MathematicalProgram,
) -> None:
    all_vars = prog.decision_variables()
    for i in range(manager.params.M):
        for j in range(manager.params.n - 1):
            t_ijl_exp = manager.compute_t_ijl_exp(
                t_i_var=manager.get_t_i_var(all_vars, i),
                j=j,
                l=0,
            )
            sigma_i = manager.compute_sigma_i_exp(
                c_x_i_vars=manager.get_c_x_i_vars(all_vars, i),
                c_y_i_vars=manager.get_c_y_i_vars(all_vars, i),
                c_theta_i_vars=manager.get_c_theta_i_vars(all_vars, i),
                t_exp=t_ijl_exp,
                derivative=0,
            )
            assert isinstance(sigma_i, np.ndarray)
            assert sigma_i.shape == (3,)

    # Test with actual values.
    c_x_i_vars = rng.rand(2 * manager.params.h)
    c_y_i_vars = rng.rand(2 * manager.params.h)
    c_theta_i_vars = rng.rand(2 * manager.params.h)
    t_exp = 0.0

    basis_vector = np.zeros((2 * manager.params.h,), dtype=np.float64)
    basis_vector[0] = 1.0

    sigma_i = manager.compute_sigma_i_exp(
        c_x_i_vars=c_x_i_vars,
        c_y_i_vars=c_y_i_vars,
        c_theta_i_vars=c_theta_i_vars,
        t_exp=t_exp,
    )
    np.testing.assert_allclose(sigma_i[0], c_x_i_vars @ basis_vector, atol=1e-12)
    np.testing.assert_allclose(sigma_i[1], c_y_i_vars @ basis_vector, atol=1e-12)
    np.testing.assert_allclose(sigma_i[2], c_theta_i_vars @ basis_vector, atol=1e-12)

    basis_vector = np.zeros((2 * manager.params.h,), dtype=np.float64)
    basis_vector[1] = 1.0

    sigma_i = manager.compute_sigma_i_exp(
        c_x_i_vars=c_x_i_vars,
        c_y_i_vars=c_y_i_vars,
        c_theta_i_vars=c_theta_i_vars,
        t_exp=t_exp,
        derivative=1,
    )
    np.testing.assert_allclose(sigma_i[0], c_x_i_vars @ basis_vector, atol=1e-12)
    np.testing.assert_allclose(sigma_i[1], c_y_i_vars @ basis_vector, atol=1e-12)
    np.testing.assert_allclose(sigma_i[2], c_theta_i_vars @ basis_vector, atol=1e-12)


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
