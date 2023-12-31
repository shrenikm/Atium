import inspect
from typing import Optional

import numpy as np

from common.custom_types import AttrsConverterFunc, AttrsValidatorFunc, NpArrf64


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
        def _num_args_validator(value) -> None:
            assert inspect.isfunction(value)
            sig = inspect.signature(value)
            num_args = len(sig.parameters)

            assert num_min_args <= num_args <= num_max_args

        return _num_args_validator
