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


# Numpy types.
NpVectorNf64 = Annotated[npt.NDArray[f64], Literal["N"]]
NpVector1f64 = Annotated[npt.NDArray[f64], Literal["1"]]
NpVector2f64 = Annotated[npt.NDArray[f64], Literal["2"]]
NpVector3f64 = Annotated[npt.NDArray[f64], Literal["3"]]
NpScalarf64 = Union[NpVector1f64, float]

NpScalarOrVectorNf64 = Union[NpScalarf64, NpVectorNf64]

NpMatrixMNf64 = Annotated[npt.NDArray[f64], Literal["M, N"]]
NpMatrixNNf64 = Annotated[npt.NDArray[f64], Literal["N, N"]]
NpMatrixN2f64 = Annotated[npt.NDArray[f64], Literal["N, 2"]]

NpTensorLMNf64 = Annotated[npt.NDArray[f64], Literal["L, M, N"]]
NpTensorMN3ui8 = Annotated[npt.NDArray[ui8], Literal["M, N, 3"]]

# JAX types.
# TODO: Use jpt.Array here after updating to newer JAX
# jpt.ArrayLike includes np arrays as well so there is a bit of an overlap
JpVectorNf64 = Annotated[jpt.ArrayLike, Literal["N"]]
JpVector1f64 = Annotated[jpt.ArrayLike, Literal["1"]]
JpVector2f64 = Annotated[jpt.ArrayLike, Literal["2"]]
JpVector3f64 = Annotated[jpt.ArrayLike, Literal["3"]]
JpScalarf64 = JpVector1f64

JpScalarOrVectorNf64 = Union[JpScalarf64, JpVectorNf64]

JpMatrixMNf64 = Annotated[jpt.ArrayLike, Literal["M, N"]]
JpMatrixNNf64 = Annotated[jpt.ArrayLike, Literal["N, N"]]
JpMatrixN2f64 = Annotated[jpt.ArrayLike, Literal["N, 2"]]

JpTensorLMNf64 = Annotated[jpt.ArrayLike, Literal["L, M, N"]]
JpTensorMN3ui8 = Annotated[jpt.ArrayLike, Literal["M, N, 3"]]

# Combined types.
VectorNf64 = Union[NpVectorNf64, JpVectorNf64]
Vector1f64 = Union[NpVector1f64, JpVector1f64]
Vector2f64 = Union[NpVector2f64, JpVector2f64]
Vector3f64 = Union[NpVector3f64, JpVector3f64]
Scalarf64 = Union[NpScalarf64, JpScalarf64]

ScalarOrVectorNf64 = Union[NpScalarOrVectorNf64, JpScalarOrVectorNf64]

MatrixMNf64 = Union[NpMatrixMNf64, JpMatrixMNf64]
MatrixNNf64 = Union[NpMatrixNNf64, JpMatrixNNf64]
MatrixN2f64 = Union[NpMatrixN2f64, JpMatrixN2f64]

TensorLMNf64 = Union[NpTensorLMNf64, JpTensorLMNf64]
TensorMN3ui8 = Union[NpTensorMN3ui8, JpTensorMN3ui8]


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
