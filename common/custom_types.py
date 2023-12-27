from typing import Any, Callable, Tuple

import nptyping as npt

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

# Geometry
PointXYVector = Arr[Shape["2"], f64]
PolygonXYArray = Arr[Shape["N, 2"], f64]
PointXYArray = Arr[Shape["2"], f64]
SizeXYVector = Arr[Shape["2"], f64]
CoordinateXY = Tuple[int, int]

# Kinematics/dynamics/control
AMatrix = Arr[Shape["Nx, Nx"], f64]
BMatrix = Arr[Shape["Nx, Nu"], f64]
StateVector = Arr[Shape["Nx"], f64]
ControlVector = Arr[Shape["Nu"], f64]
VelocityXYVector = Arr[Shape["2"], f64]

VelocityXYArray = Arr[Shape["N, 2"], f64]
StateTrajectoryArray = Arr[Shape["N, Nx"], f64]
ControlTrajectoryArray = Arr[Shape["N, Nu"], f64]

# Optimization
CostVector = Arr[Shape["N"], f64]
CostMatrix = Arr[Shape["N,M"], f64]
