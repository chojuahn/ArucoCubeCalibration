#include "PlotUtils.hpp"

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "GeometryUtils.hpp"


namespace Util
{
void drawProjectedCorners(
    cv::Mat& bgr,
    const Eigen::RowVector<double, 6>& cube_to_camera_rtvec,
    const std::vector<Marker3d>& markers_cube,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    const cv::Scalar& color,
    int marker_type)
{
    const auto cube_to_camera = rtToMat4x4(cube_to_camera_rtvec);
    std::vector<cv::Point3d> corners_camera_cv;
    for (const auto& marker_cube : markers_cube) {
        const auto marker_camera = transform4x4(cube_to_camera, marker_cube);
        for (size_t r = 0; r < marker_camera.rows(); ++r)
            corners_camera_cv.emplace_back(marker_camera(r, 0), marker_camera(r, 1), marker_camera(r, 2));
    }
    
    std::vector<cv::Point2d> corners_image_cv;
    cv::projectPoints(corners_camera_cv, cv::Mat::zeros(3, 1, CV_64F), cv::Mat::zeros(3, 1, CV_64F), camera_matrix, dist_coeffs, corners_image_cv);
    for(const auto& corner_image_cv : corners_image_cv) {
        cv::drawMarker(bgr, corner_image_cv, color, marker_type, 20, 1);
    }
}

void drawMarkerLines(
    cv::Mat& bgr,
    const Eigen::RowVector<double, 6>& cube_to_camera_rtvec,
    const std::vector<Marker3d>& markers_cube,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    const cv::Scalar& color)
{
    const auto cube_to_camera = rtToMat4x4(cube_to_camera_rtvec);
    std::vector<std::vector<cv::Point3d>> markers_camera_cv;
    for (const auto& marker_cube : markers_cube) {
        const auto marker_camera = transform4x4(cube_to_camera, marker_cube);
        auto& marker_camera_cv = markers_camera_cv.emplace_back(std::vector<cv::Point3d>{});
        for (size_t r = 0; r < marker_camera.rows(); ++r)
            marker_camera_cv.emplace_back(marker_camera(r, 0), marker_camera(r, 1), marker_camera(r, 2));
    }

    for (const auto& marker_camera_cv : markers_camera_cv) {
        std::vector<cv::Point2d> marker_image_cv;
        cv::projectPoints(marker_camera_cv, cv::Mat::zeros(3, 1, CV_64F), cv::Mat::zeros(3, 1, CV_64F), camera_matrix, dist_coeffs, marker_image_cv);
        cv::line(bgr, marker_image_cv[0], marker_image_cv[1], color, 1, cv::LINE_4);
        cv::line(bgr, marker_image_cv[1], marker_image_cv[2], color, 1, cv::LINE_4);
        cv::line(bgr, marker_image_cv[2], marker_image_cv[3], color, 1, cv::LINE_4);
        cv::line(bgr, marker_image_cv[3], marker_image_cv[0], color, 1, cv::LINE_4);
    }
}
}
