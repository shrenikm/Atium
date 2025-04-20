"""
Implementation of the paper: Universal Trajectory Optimization Framework for  Differential Drive Robot Class
See: https://arxiv.org/abs/2409.07924
"""

import attr
import numpy as np
from pydrake.solvers import MathematicalProgram


@attr.frozen
class UnitoParams:
    """
    Parameters for Unito.
    Includes problem formulation and optimization params.
    """

    # Basis beta will be of degree 2*h-1
    h: float
    # Number of segments
    M: int
    # Number of sampling intervals
    n: int


@attr.define
class Unito:
    params: UnitoParams

    # Optimization variables
    _prog: MathematicalProgram = attr.ib(init=False)
    _var_c: np.ndarray = attr.ib(init=False)
    _var_t: np.ndarray = attr.ib(init=False)

    @_prog.default
    def _init_prog(self):
        return MathematicalProgram()

    @_var_c.default
    def _init_var_c(self):
        return self._prog.NewContinuousVariables(2 * self.params.h * self.params.M, 2, "c")

    @_var_t.default
    def _init_var_t(self):
        return self._prog.NewContinuousVariables(self.params.M, 1, "t")


params = UnitoParams(h=2, M=3, n=4)
unito = Unito(params)
