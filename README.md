# ArUco Cube Camera Calibration

A toolkit for ArUco cube calibration with/without a calibrated camera that accommodates imperfect cube dimensions. This means the cube doesn't need to be exactly the intended size, nor do its surfaces need to be perfectly flat.

It is a process of optimizing local coordinates of the cube's ArUco corners, with optional camera intrinsics calibration if not provided.

The algorithm is similar to conventional bundle adjustment, which simultaneously optimizes camera intrinsics/extrinsics and 3D coordinates of local features.

Unlike traditional methods that use random feature detection and matching, this approach uses known features (ArUco markers) on each face of the cube to eliminate ambiguities.

The optimization uses dlib's L-BFGS-B optimizer, which can be replaced with more advanced optimizers like Ceres-solver.

The calibrated ArUco cube serves as a reference coordinate system for camera pose estimation using all 6 faces of the cube. This is useful for tasks like 3D reconstruction and multi-frame scene localization in controlled environments.

## Build Instructions

### Prerequisites

- CMake (3.10 or higher)
- ffmpeg
- ArUco cube with 6 faces of ArUco markers
  - 5x5 resolution of ArUco markers with ids 0 to 5 (configurable via --marker-res)
  - Making ArUco cube doesn't require any special tools, like 3D printer, just a few sheets of ArUco markers and boards to form the cube.
  - Rough size of the cube and the exact ArUco marker length are required to perform the calibration.
  - Current implementation of initial markers placement is hardcoded such as
    - +z: id 0,
    - -y: id 1,
    - +x: id 2,
    - -z: id 3,
    - -x: id 4,
    - +y: id 5
  - as well as initial marker orientation can be found in `getInitialCornersInCube()` function to be modified if needed.
  - Note: Incorrect marker placement compared to initial marker coordinates will lead to longer time for solution to converge, and it can be stuck in local minima.

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
  --marker-res STRING    Marker resolution (default: "5x5")
  --cam-json STRING      Path to initial camera intrinsics JSON file (optional)
  --replay STRING        Path to replay video file (optional)
  -h, --help            Print usage information
```

#### Example Usage

```bash
./calibrate_aruco_cube_camera --nsamples 100 --marker-length 0.04 --marker-res 5x5
```

#### What it does

1. Processes camera stream to detect ArUco markers (press 'q' or `ESC` to exit)
2. Optimizes local coordinates of cube corners and camera intrinsics/extrinsics (extrinsics per frame)
3. Saves calibration results to a JSON file in std::filesystem::current_path()
```
camera_matrix: 3x3 array
distortion_coefs: 1x5 array
markers_in_cube: 6x4x2 array
```
4. Generates a visualization video showing initial vs optimized corner projections

https://github.com/user-attachments/assets/d3f0a6fa-4347-4755-aa18-21f190fc3202

- Red diamonds: Detected 2D corners used for pose estimation (only one marker being used for testing)
- Yellow crosses: Initial guess of local corners
- Green crosses: Optimized local corners
- Green lines: Marker boundaries
- Coordinate axes: Estimated cube pose

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
