from common.custom_types import ScalarOrVectorNf64, VectorOrMatrixNf64


def assert_gradient_sizes(
    x0: VectorNf64,
    x_grad: VectorOrMatrixNf64,
    num_variables: int,
) -> None:
    if x0.size == 1:
        # Single constraint
        assert x0.ndim == 0
        assert x_grad.ndim == 1
        assert x_grad.size == num_variables
    else:
        # Multiple constraints.
        assert x0.ndim == 1
        assert x_grad.ndim == 2
        assert x_grad.shape == (x0.size, num_variables)
