import attr
import numpy as np
import pytest

from atium.core.utils.attrs_utils import AttrsValidators


def test_scalar_bounding_box_validator() -> None:
    # No bounds.
    @attr.define
    class A:
        val: int = attr.ib(validator=AttrsValidators.scalar_bounding_box_validator())

    A(5)
    A(-5)

    # Lower limit.
    @attr.define
    class A:
        val: float = attr.ib(
            validator=AttrsValidators.scalar_bounding_box_validator(
                min_value=0,
                inclusive=False,
            )
        )

    A(0.1)
    A(5)

    with pytest.raises(ValueError):
        A(0)

    # Upper limit.
    @attr.define
    class A:
        val: float = attr.ib(
            validator=AttrsValidators.scalar_bounding_box_validator(
                max_value=10,
                inclusive=True,
            )
        )

    A(10)
    A(-10)

    with pytest.raises(ValueError):
        A(10.1)


def test_array_2d_validator() -> None:
    # No desired dtype.
    @attr.define
    class A:
        val: np.ndarray = attr.ib(validator=AttrsValidators.array_2d_validator())

    A(np.array([[1, 2], [3, 4]], dtype=np.int32))
    A(np.array([[1, 2], [3, 4]], dtype=np.float64))

    with pytest.raises(ValueError):
        A(np.array([1, 2, 3, 4], dtype=np.int32))

    # With desired dtype.
    @attr.define
    class A:
        val: np.ndarray = attr.ib(validator=AttrsValidators.array_2d_validator(desired_dtype=np.uint8))

    A(np.array([[1, 2], [3, 4]], dtype=np.uint8))

    with pytest.raises(ValueError):
        A(np.array([[1, 2], [3, 4]], dtype=np.int32))


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
