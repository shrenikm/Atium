from typing import Any, Optional
import numpy as np
import nptyping as npt
from common.custom_types import Arr, AttrsConverterFunc


class AttrsConverters:
    @classmethod
    def np_f64_converter(
        cls,
        precision: Optional[int] = None,
    ) -> AttrsConverterFunc:
        def _np_array_converter(value) -> Arr:
            np_value = np.array(value, dtype=np.float64)
            if precision is not None:
                np_value = np_value.round(precision)
            return np_value

        return _np_array_converter
