import inspect
from typing import Optional

import numpy as np

from atium.core.utils.custom_exceptions import AtiumAttributeError
from atium.core.utils.custom_types import AttrsConverterFunc, AttrsValidatorFunc, NpArrf64

# TODO: Tests for these.


class AttrsConverters:
    @classmethod
    def np_f64_converter(
        cls,
        precision: Optional[int] = None,
    ) -> AttrsConverterFunc:
        def _np_array_converter(value) -> NpArrf64:
            np_value = np.array(value, dtype=np.float64)
            if precision is not None:
                np_value = np_value.round(precision)
            return np_value

        return _np_array_converter


class AttrsValidators:
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
                raise AtiumAttributeError(
                    f"Number of arguments to {attribute} needs to be between {num_min_args} and {num_max_args}"
                )

        return _num_args_validator
