from numbers import Number
from typing import (
    Annotated,
    Any,
    Callable,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)

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

VectorN = Annotated[Arr[DType], Literal["N"]]
Vector1 = Annotated[Arr[DType], Literal["1"]]
Vector2 = Annotated[Arr[DType], Literal["2"]]

Scalar = Union[Vector1[f64], Vector1[i64], float]

ScalarOrVectorN = Union[Scalar, VectorN]

MatrixMN = Annotated[Arr[DType], Literal["M, N"]]
MatrixNN = Annotated[Arr[DType], Literal["N, N"]]
MatrixN2 = Annotated[Arr[DType], Literal["N, 2"]]

TensorLMN = Annotated[Arr[DType], Literal["L, M, N"]]
TensorMN3 = Annotated[Arr[DType], Literal["M, N, 3"]]
VectorNf64 = VectorN[f64]
Vector2f64 = Vector2[f64]
MatrixMNf64 = MatrixMN[f64]
MatrixNNf64 = MatrixNN[f64]
MatrixN2f64 = MatrixN2[f64]
TensorLMNf64 = TensorLMN[f64]
TensorMN3ui8 = TensorMN3[ui8]


# Calculus stuff
DerivativeVector = VectorNf64  # Generic float vector
# Time derivative: dx/dt = f(t, x)
TimeDerivativeFn = Callable[[float, VectorNf64], DerivativeVector]

# Function that takes in a vector and returns a scalar.
ScalarInputScalarOutputFnWithoutParams = Callable[[Scalar], Scalar]
ScalarInputScalarOutputFnWithParams = Callable[[Scalar, Any], Scalar]
ScalarInputScalarOutputFn = Union[
    ScalarInputScalarOutputFnWithoutParams, ScalarInputScalarOutputFnWithParams
]

# Function that takes in a vector and returns a vector.
ScalarInputVectorOutputFnWithoutParams = Callable[[Scalar], VectorNf64]
ScalarInputVectorOutputFnWithParams = Callable[[Scalar, Any], VectorNf64]
ScalarInputVectorOutputFn = Union[
    ScalarInputVectorOutputFnWithoutParams, ScalarInputVectorOutputFnWithParams
]

# Function that takes in a vector and returns a scalar.
VectorInputScalarOutputFnWithoutParams = Callable[[VectorNf64], Scalar]
VectorInputScalarOutputFnWithParams = Callable[[VectorNf64, Any], Scalar]
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

if __name__ == "__main__":

    def g(fn: OptimizationFn):
        print(fn)
        print("g")

    def f(x: float) -> float:
        return x * 2.0

    print(ScalarOrVectorN)
    g(fn=f)
