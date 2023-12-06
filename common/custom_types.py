import nptyping as npt

Arr = npt.NDArray
Shape = npt.Shape
f64 = npt.Float64

# Geometry
PointXYVector = Arr[Shape["2"], f64]
Polygon2DArray = Arr[Shape["n, 2"], f64]
PointXYArray = Arr[Shape["2"], f64]

# Kinematics/dynamics/control
AMatrix = Arr[Shape["nx, nx"], f64]
BMatrix = Arr[Shape["nx, nu"], f64]
StateVector = Arr[Shape["nx"], f64]
ControlVector = Arr[Shape["nu"], f64]
VelocityXYVector = Arr[Shape["2"], f64]

VelocityXYArray = Arr[Shape["N, 2"], f64]
StateTrajectoryArray = Arr[Shape["N, nx"], f64]
ControlTrajectoryArray = Arr[Shape["N, nu"], f64]

# Optimization
CostVector = Arr[Shape["n"], f64]

