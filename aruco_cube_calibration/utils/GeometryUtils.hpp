#pragma once
#include <Eigen/Core>
#include <opencv2/core/mat.hpp>

#include "Types.hpp"


namespace Util
{
template <typename SizeType>
void forEachIdx(const SizeType loop_size, const std::function<void(const SizeType)>& func);

Marker3d rotateAboutX(const Marker3d& marker3d, const double angle);

Marker3d rotateAboutY(const Marker3d& marker3d, double angle);

Marker3d rotateAboutZ(const Marker3d& marker3d, double angle);

std::tuple<cv::Mat, cv::Mat> toCvCameraIntrinsics(const Eigen::Matrix3d& camera_matrix, const Eigen::Vector<double, 5>& distortion_coefs);

std::tuple<Eigen::Matrix3d, Eigen::Vector<double, 5>> toEigenCameraIntrinsics(const cv::Mat& camera_matrix_cv, const cv::Mat& distortion_coefs_cv);

Eigen::Matrix4d rtToMat4x4(const Eigen::RowVector<double, 6>& rvec);

Eigen::MatrixX2d transform3x3(const Eigen::Matrix3d& mat, const Eigen::MatrixXd& pts);

Eigen::MatrixX3d transform4x4(const Eigen::Matrix4d& mat, const Eigen::MatrixXd& pts);

Eigen::Matrix3d rodrigues(const Eigen::Vector3d& rvec);

std::vector<size_t> getSpatiallyDistributedFrameIndices(const std::vector<Eigen::Vector2d>& frame_markers, size_t n_samples);

std::tuple<Eigen::Vector3d, Eigen::Vector3d> getPointcloudPose(
    const Eigen::MatrixX3d& pointcloud_local,
    const Eigen::MatrixX2d& pointcloud_image,
    const Eigen::Matrix3d& camera_matrix,
    const Eigen::Vector<double, 5>& distortion_coefs);

std::array<Marker3d, 6> alignCornersToOrthogonalFrame(const std::array<Marker3d, 6>& corners_cube_6x4x3);
}
