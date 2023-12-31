from typing import Optional

import numpy as np

from common.custom_types import NpArrf64, AttrsConverterFunc


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
