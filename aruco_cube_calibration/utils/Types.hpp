#pragma once
#include <optional>

#include <Eigen/Core>

typedef Eigen::Matrix<double, 4, 2> Marker2d;
typedef Eigen::Matrix<double, 4, 3> Marker3d;
typedef Eigen::Matrix<double, -1, 6> MatrixX6d;
typedef std::optional<Marker2d> Marker2dOpt;
