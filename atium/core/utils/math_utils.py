import numpy as np

from atium.core.utils.custom_types import MatrixNNf64


def assert_matrix_positive_definite(mat: MatrixNNf64) -> None:
    assert mat.ndim == 2
    assert mat.shape[0] == mat.shape[1]
    assert np.all(np.linalg.eigvals(mat) > 0)


def assert_matrix_positive_semidefinite(mat: MatrixNNf64) -> None:
    assert mat.ndim == 2
    assert mat.shape[0] == mat.shape[1]
    assert np.all(np.linalg.eigvals(mat) >= 0)
