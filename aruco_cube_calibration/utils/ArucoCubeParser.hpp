#pragma once
#include <filesystem>
#include <map>

#include <opencv2/core/types.hpp>
#include "opencv2/videoio.hpp"
#include <opencv2/objdetect/aruco_detector.hpp>
#include <Eigen/Core>

#include "Types.hpp"


namespace Util
{
class ArucoCubeParser
{
public:
    ArucoCubeParser(const std::filesystem::path& replay_video_path, const size_t min_valid_n_markers, const double marker_length, const std::string& marker_res);

    ~ArucoCubeParser();

    const cv::Size& getImageSize() const;

    cv::Mat getBgr(const size_t frame_idx);

    std::tuple<Eigen::Matrix3d, Eigen::Vector<double, 5>, std::vector<std::vector<Marker2dOpt>>, std::vector<size_t>>
    processFrames(const std::optional<Eigen::Matrix3d>& camera_matrix_opt, const std::optional<Eigen::Vector<double, 5>>& distortion_coefs_opt);

private:
    std::tuple<std::vector<Marker2d>, std::vector<int>> getUniqueMarkers(
        const std::vector<Marker2d>& markers_image,
        const std::vector<int>& ids,
        const size_t id_limit,
        const double tilt_degree_limit,
        const double xy_ratio_limit);

    bool isBlurry(const cv::Mat& gray, const std::vector<Marker2d>& markers_image) const;

    // Config
    const size_t m_min_valid_n_markers;
    const double m_marker_length;
    const size_t m_id_limit = 6;
    const double m_blur_threshold = 100.0;

    // Capture
    cv::VideoCapture m_cap;
    cv::Size m_image_size;

    // ArUco
    cv::aruco::ArucoDetector m_detector;

    // Stored results
    std::vector<std::vector<Marker2dOpt>> m_markers_image_ided_framed;
    std::vector<size_t> m_frame_numbers;
    std::map<size_t, cv::Mat> m_bgrs;
};
}
