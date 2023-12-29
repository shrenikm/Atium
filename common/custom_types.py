from typing import Annotated, Any, Callable, List, Literal, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt

DType = TypeVar("DType", bound=npt.DTypeLike)

# File/directory types
FileName = str
DirectoryName = str
FilePath = str
DirectoryPath = str
OutputVideoName = str
OutputVideoPath = str

# Attrs stuff
AttrsConverterFunc = Callable[[Any], Any]

# Indices
Index2D = Tuple[int, int]
Indices2D = List[Index2D]
Index3D = Tuple[int, int, int]
Indices3D = List[Index3D]

# General Array stuff
Arr = npt.NDArray
Scalar = Annotated[Arr[DType], Literal["1"]]

VectorN = Annotated[Arr[DType], Literal["N"]]
Vector2 = Annotated[Arr[DType], Literal["2"]]

MatrixMN = Annotated[Arr[DType], Literal["M, N"]]
MatrixNN = Annotated[Arr[DType], Literal["N, N"]]
MatrixN2 = Annotated[Arr[DType], Literal["N, 2"]]

Tensor3D = Annotated[Arr[DType], Literal["L, M, N"]]
TensorMN3 = Annotated[Arr[DType], Literal["M, N, 3"]]
f64 = np.float64
ui8 = np.uint8
VectorNf64 = VectorN[f64]
Vector2f64 = Vector2[f64]
MatrixMNf64 = MatrixMN[f64]
MatrixNNf64 = MatrixNN[f64]
MatrixN2f64 = MatrixN2[f64]
Tensor3Df64 = Tensor3D[f64]
TensorMN3ui8 = TensorMN3[ui8]


# Calculus stuff
DerivativeVector = VectorNf64  # Generic float vector
# Time derivative: dx/dt = f(t, x)
TimeDerivativeFn = Callable[[float, VectorNf64], DerivativeVector]
# Function that takes in a vector and returns a scalar.
VectorInputScalarOutputFn = Callable[[VectorNf64], Scalar]
# Function that takes in a vector and returns a vector.
VectorInputVectorOutputFn = Callable[[VectorNf64], VectorNf64]
# Gradient for a function that takes a vector and returns the gradient vector.
# This is for a vector input scalar output function.
VectorInputVectorOutputGradFn = Callable[[VectorNf64], VectorNf64]
# Hessian for a function that takes a vector and returns the hessian matrix.
# This is for a vector input scalar output function.
VectorInputMatrixOutputHessFn = Callable[[VectorNf64], MatrixMNf64]
# Gradient for a function that takes a vector and returns the gradient matrix (jacobian).
# This is for a vector input vector output function.
VectorInputMatrixOutputGradFn = Callable[[VectorNf64], MatrixMNf64]
# Hessian for a function that takes a vector and returns the hessian tensor.
# This is for a vector input vector output function.
VectorInputTensorOutputHessFn = Callable[[VectorNf64], Tensor3Df64]

# Geometry
AnglesRad = VectorNf64
AngleOrAnglesRad = Union[float, AnglesRad]
PointXYVector = Vector2f64
PolygonXYArray = MatrixN2f64
PointXYArray = MatrixN2f64
SizeXYVector = Vector2f64
CoordinateXY = Tuple[float, float]


# Kinematics/dynamics/control
AMatrix = MatrixNNf64
BMatrix = MatrixMNf64
StateVector = VectorNf64
StateDerivativeVector = VectorNf64
ControlInputVector = VectorNf64
VelocityXYVector = Vector2f64
# State derivative: dx/dt = f(x, u). We dont' explicitly include t.
StateDerivativeFn = Callable[[StateVector, ControlInputVector], StateDerivativeVector]


VelocityXYArray = MatrixN2f64
StateTrajectoryArray = MatrixMNf64
ControlTrajectoryArray = MatrixMNf64

# Optimization
CostVector = VectorNf64
CostMatrix = MatrixMNf64

# Visualization
BGRColor = Tuple[int, int, int]
Img3Channel = TensorMN3ui8
