from typing import Any, Callable, List, Tuple, TypeVar, Union

import nptyping as npt
import numpy as np

# File/directory types
FileName = str
DirectoryName = str
FilePath = str
DirectoryPath = str
OutputVideoName = str
OutputVideoPath = str

# Attrs stuff
AttrsConverterFunc = Callable[[Any], Any]


# Array stuff
Arr = npt.NDArray
Shape = npt.Shape
f64 = npt.Float64
ui8 = npt.UInt8
Index2D = Tuple[int, int]
Indices2D = List[Index2D]
Index3D = Tuple[int, int, int]
Indices3D = List[Index3D]

# Calculus stuff
Scalar = Arr[Shape["1"], f64]  # TODO: Better type maybe?
Vector = Arr[Shape["N"], f64]  # Generic float vector
Matrix = Arr[Shape["M, N"], f64]  # Generic float matrix
Tensor3D = Arr[Shape["L, M, N"], f64]  # Generic float 3d tensor

DerivativeVector = Arr[Shape["N"], f64]  # Generic float vector
# Time derivative: dx/dt = f(t, x)
TimeDerivativeFn = Callable[[float, Vector], DerivativeVector]
# Function that takes in a vector and returns a scalar.
VectorInputScalarOutputFn = Callable[[Vector], Scalar]
# Function that takes in a vector and returns a vector.
VectorInputVectorOutputFn = Callable[[Vector], Vector]
# Gradient for a function that takes a vector and returns the gradient vector.
# This is for a vector input scalar output function.
VectorInputVectorOutputGradFn = Callable[[Vector], Vector]
# Hessian for a function that takes a vector and returns the hessian matrix.
# This is for a vector input scalar output function.
VectorInputMatrixOutputHessFn = Callable[[Vector], Matrix]
# Gradient for a function that takes a vector and returns the gradient matrix (jacobian).
# This is for a vector input vector output function.
VectorInputMatrixOutputGradFn = Callable[[Vector], Matrix]
# Hessian for a function that takes a vector and returns the hessian tensor.
# This is for a vector input vector output function.
VectorInputTensorOutputHessFn = Callable[[Vector], Tensor3D]

# Geometry
AnglesRad = Arr[Shape["N"], f64]
AngleOrAnglesRad = Union[float, AnglesRad]
PointXYVector = Arr[Shape["2"], f64]
PolygonXYArray = Arr[Shape["N, 2"], f64]
PointXYArray = Arr[Shape["2"], f64]
SizeXYVector = Arr[Shape["2"], f64]
CoordinateXY = Tuple[float, float]


# Kinematics/dynamics/control
AMatrix = Arr[Shape["Nx, Nx"], f64]
BMatrix = Arr[Shape["Nx, Nu"], f64]
StateVector = Arr[Shape["Nx"], f64]
StateDerivativeVector = Arr[Shape["Nx"], f64]
ControlInputVector = Arr[Shape["Nu"], f64]
VelocityXYVector = Arr[Shape["2"], f64]
# State derivative: dx/dt = f(x, u). We dont' explicitly include t.
StateDerivativeFn = Callable[[StateVector, ControlInputVector], StateDerivativeVector]


VelocityXYArray = Arr[Shape["N, 2"], f64]
StateTrajectoryArray = Arr[Shape["N, Nx"], f64]
ControlTrajectoryArray = Arr[Shape["N, Nu"], f64]

# Optimization
CostVector = Arr[Shape["N"], f64]
CostMatrix = Arr[Shape["N,M"], f64]

# Visualization
BGRColor = Tuple[int, int, int]
Img3Channel = Arr[Shape["M, N, 3"], ui8]
