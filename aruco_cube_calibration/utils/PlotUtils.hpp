#pragma once
#include <Eigen/Core>
#include <opencv2/core/types.hpp>

#include "Types.hpp"


namespace Util
{
void drawProjectedCorners(
    cv::Mat& bgr,
    const Eigen::RowVector<double, 6>& cube_to_camera_rtvec,
    const std::vector<Marker3d>& markers_cube,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    const cv::Scalar& color,
    int marker_type);

void drawMarkerLines(
    cv::Mat& bgr,
    const Eigen::RowVector<double, 6>& cube_to_camera_rtvec,
    const std::vector<Marker3d>& markers_cube,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    const cv::Scalar& color);
}
