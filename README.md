# ArUco Cube Camera Calibration

A toolkit for ArUco cube calibration with/without calibrated camera. ArUco cube calibration is a process of optimizing local coordinates of the cube's ArUco corners, given the camera intrinsics and poses. If not, it will also calibrate the camera intrinsics.

The algorithm is similar to traditional bundle adjustment, which optimizes camera intrinsics/extrinsics and 3D coordinates of local features, although this one is with known features on each face of the cube (ArUco markers).

Optimization is done using dlib's L-BFGS-B optimizer, which can be replaced with more advanced optimizers, e.g. Ceres-solver.

The calibrated ArUco cube can be used as a reference coordinate system for camera pose estimation from all 6 faces of the cube, which is useful for tasks like 3D reconstruction, multi-frame scene registration, etc.

## Build Instructions

### Prerequisites

- CMake (3.10 or higher)
- ffmpeg
- ArUco cube with 6 faces of ArUco markers
  - 5x5 resolution of ArUco markers with ids 0 to 5
  - Making ArUco cube doesn't require any special tools, like 3D printer, just a few sheets of ArUco markers and boards to form the cube.
  - Rough size of the cube and the exact ArUco marker length are required to perform the calibration.
  - Current implementation of markers placement is hardcoded such as
    - +z: id 0,
    - -y: id 1,
    - +x: id 2,
    - -z: id 3,
    - -x: id 4,
    - +y: id 5
  - Note: This can be changed in the code, or arg parser can be extended to read initial local cube corners from the user

### Building

1. Clone the repository:
```bash
git clone <repository-url>
```

2. Create and navigate to build directory:
```bash
mkdir build && cd build
```

3. Configure and build:
```bash
cmake ..
make
```

## Applications

### calibrate_aruco_cube_camera

This application performs ArUco cube calibration assuming the camera is not calibrated. It optimizes Aruco cube corner positions as well as camera intrinsics (focal length, principal point, distortion coefficients) using corner detections across multiple frames.

Cube specific sorting algorithm is also implemented to find the best distribution of frames for each combination of corners.

The result is a JSON file containing the optimized camera intrinsics and cube corner positions, as well as a video showing the initial vs optimized corner projections.

#### Arguments

```
Options:
  --nsamples INT         Number of samples to use for calibration
  --marker-length FLOAT  Length of ArUco marker in meters
  --cam-json STRING      Path to initial camera intrinsics JSON file (optional)
  --replay STRING        Path to video replay folder (optional)
  -h, --help            Print usage information
```

#### Example Usage

```bash
./calibrate_aruco_cube_camera --nsamples 100 --marker-length 0.04
```

#### Output

The application:
1. Processes camera stream to detect ArUco markers
2. Optimizes cube corner positions and camera intrinsics
3. Saves calibration results to a JSON file
4. Generates a visualization video showing initial vs optimized corner projections

The output includes:
- Camera matrix (intrinsics)
- Distortion coefficients
- Optimized cube corner positions
- Visualization video showing the calibration results

## File Structure

```
.
├── aruco_cube_calibration/
│   ├── calibrate_aruco_cube_camera/
│   │   └── calibrate_aruco_cube_camera.cpp
│   └── utils/
│       ├── ArucoCubeParser.hpp/cpp
│       ├── GeometryUtils.hpp/cpp
│       ├── Json.hpp/cpp
│       ├── PlotUtils.hpp/cpp
│       └── Types.hpp
├── thirdparty/CMakeLists.txt
└── CMakeLists.txt
```

## Dependencies

The project uses several third-party libraries:
- OpenCV (with contrib modules for ArUco detection)
- Eigen3 (for linear algebra operations)
- dlib (for optimization)
- cxxopts (for command line argument parsing)
- nlohmann/json (for JSON parsing)
- ffmpeg (for video processing and result visualization)

These dependencies are managed using CMake's FetchContent module, except for ffmpeg, which should be installed separately.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
