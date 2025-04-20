"""
Implementation of the paper: Universal Trajectory Optimization Framework for  Differential Drive Robot Class
See: https://arxiv.org/abs/2409.07924
"""

import attr


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


@attr.s
class Unito:
    params: UnitoParams
