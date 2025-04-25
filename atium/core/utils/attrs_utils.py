import inspect

import numpy as np

from atium.core.utils.custom_types import AttrsConverterFunc, AttrsValidatorFunc, NpArrf64

# TODO: Tests for these.


class AttrsConverters:
    @classmethod
    def np_f64_converter(
        cls,
        precision: int | None = None,
    ) -> AttrsConverterFunc:
        def _np_array_converter(value) -> NpArrf64:
            np_value = np.array(value, dtype=np.float64)
            if precision is not None:
                np_value = np_value.round(precision)
            return np_value

        return _np_array_converter


class AttrsValidators:
    @classmethod
    def scalar_bounding_box_validator(
        cls,
        min_value: float = -np.inf,
        max_value: float = np.inf,
        inclusive: bool = True,
    ) -> AttrsValidatorFunc:
        def _scalar_bounding_box_validator(instance, attribute, value) -> None:
            if not isinstance(value, (int, float, np.ndarray)):
                raise ValueError(f"Value for {attribute} must be a scalar. Got {type(value)}.")
            if isinstance(value, np.ndarray):
                if value.ndim != 0:
                    raise ValueError(f"Value for {attribute} must be a scalar. Got {value}.")
                value = value.item()
            if inclusive:
                if value < min_value or value > max_value:
                    raise ValueError(f"Value for {attribute} must be between {min_value} and {max_value}. Got {value}.")
            else:
                if value <= min_value or value >= max_value:
                    raise ValueError(f"Value for {attribute} must be between {min_value} and {max_value}. Got {value}.")

        return _scalar_bounding_box_validator

    @classmethod
    def num_args_validator(
        cls,
        num_min_args: int,
        num_max_args: int,
    ) -> AttrsValidatorFunc:
        def _num_args_validator(instance, attribute, value) -> None:
            del instance  # Cleaner to do this for type checking.
            sig = inspect.signature(value)
            num_args = len(sig.parameters)

            valid_num_args = num_min_args <= num_args <= num_max_args
            if not valid_num_args:
                raise ValueError(
                    f"Number of arguments to {attribute} needs to be between {num_min_args} and {num_max_args}"
                )

        return _num_args_validator
