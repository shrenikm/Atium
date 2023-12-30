from typing import (
    Annotated,
    Any,
    Callable,
    List,
    Literal,
    Tuple,
    TypeVar,
    Union,
)

import jax.typing as jpt
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
f64 = np.float64
i64 = np.int64
ui8 = np.uint8
Arr = npt.NDArray


VectorNf64 = Annotated[npt.NDArray[f64], Literal["N"]]
Vector1f64 = Annotated[npt.NDArray[f64], Literal["1"]]
Vector2f64 = Annotated[npt.NDArray[f64], Literal["2"]]

Scalarf64 = Union[Vector1f64, float, jpt.ArrayLike]

ScalarOrVectorNf64 = Union[Scalarf64, VectorNf64, jpt.ArrayLike]

MatrixMNf64 = Annotated[npt.NDArray[f64], Literal["M, N"]]
MatrixNNf64 = Annotated[npt.NDArray[f64], Literal["N, N"]]
MatrixN2f64 = Annotated[npt.NDArray[f64], Literal["N, 2"]]

TensorLMNf64 = Annotated[npt.NDArray[f64], Literal["L, M, N"]]
TensorMN3ui8 = Annotated[npt.NDArray[ui8], Literal["M, N, 3"]]


# Calculus stuff
DerivativeVector = VectorNf64  # Generic float vector
# Time derivative: dx/dt = f(t, x)
TimeDerivativeFn = Callable[[float, VectorNf64], DerivativeVector]

# Function that takes in a vector and returns a scalar.
ScalarInputScalarOutputFnWithoutParams = Callable[[Scalarf64], Scalarf64]
ScalarInputScalarOutputFnWithParams = Callable[[Scalarf64, Any], Scalarf64]
ScalarInputScalarOutputFn = Union[
    ScalarInputScalarOutputFnWithoutParams, ScalarInputScalarOutputFnWithParams
]

# Function that takes in a vector and returns a vector.
ScalarInputVectorOutputFnWithoutParams = Callable[[Scalarf64], VectorNf64]
ScalarInputVectorOutputFnWithParams = Callable[[Scalarf64, Any], VectorNf64]
ScalarInputVectorOutputFn = Union[
    ScalarInputVectorOutputFnWithoutParams, ScalarInputVectorOutputFnWithParams
]

# Function that takes in a vector and returns a scalar.
VectorInputScalarOutputFnWithoutParams = Callable[[VectorNf64], Scalarf64]
VectorInputScalarOutputFnWithParams = Callable[[VectorNf64, Any], Scalarf64]
VectorInputScalarOutputFn = Union[
    VectorInputScalarOutputFnWithoutParams, VectorInputScalarOutputFnWithParams
]

# Function that takes in a vector and returns a vector.
VectorInputVectorOutputFnWithoutParams = Callable[[VectorNf64], VectorNf64]
VectorInputVectorOutputFnWithParams = Callable[[VectorNf64, Any], VectorNf64]
VectorInputVectorOutputFn = Union[
    VectorInputVectorOutputFnWithoutParams, VectorInputVectorOutputFnWithParams
]

# Function that takes in a vector and returns a matrix.
VectorInputMatrixOutputFnWithoutParams = Callable[[VectorNf64], TensorLMNf64]
VectorInputMatrixOutputFnWithParams = Callable[[VectorNf64, Any], TensorLMNf64]
VectorInputMatrixOutputFn = Union[
    VectorInputMatrixOutputFnWithoutParams, VectorInputMatrixOutputFnWithParams
]

# Function that takes in a vector and returns a tensor.
VectorInputTensorOutputFnWithoutParams = Callable[[VectorNf64], MatrixMNf64]
VectorInputTensorOutputFnWithParams = Callable[[VectorNf64, Any], MatrixMNf64]
VectorInputTensorOutputFn = Union[
    VectorInputTensorOutputFnWithoutParams, VectorInputTensorOutputFnWithParams
]

# Function used in an optimization cost function. Only scalar outputs.
OptimizationCostFn = Union[
    ScalarInputScalarOutputFn,
    VectorInputScalarOutputFn,
]

# Function used in an optimization constraint.
OptimizationConstraintFn = Union[
    ScalarInputScalarOutputFn,
    ScalarInputVectorOutputFn,
    VectorInputScalarOutputFn,
    VectorInputVectorOutputFn,
]

# Function used in an optimization program -- cost or constraint.
OptimizationFn = Union[
    OptimizationCostFn,
    OptimizationConstraintFn,
]

# Gradient for a cost/constraint function that takes in a scalar/vector.
# Depending on the type, the output gradient can be a scalar, vector or a matrix (Jacobian).
OptimizationGradFn = Union[
    ScalarInputScalarOutputFn,
    ScalarInputVectorOutputFn,
    VectorInputVectorOutputFn,
    VectorInputMatrixOutputFn,
]

# Hessian for a cost/constraint function that takes in a scalar/Vector.
# Depending on the type, the output gradient can be a scalar, vector, matrix or a tensor.
OptimizationHessFn = Union[
    ScalarInputScalarOutputFn,
    ScalarInputVectorOutputFn,
    VectorInputMatrixOutputFn,
    VectorInputTensorOutputFn,
]

OptimizationGradOrHessFn = Union[OptimizationGradFn, OptimizationHessFn]


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
