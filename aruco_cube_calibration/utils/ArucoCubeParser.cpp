#include "ArucoCubeParser.hpp"

#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <limits>
#include <numeric>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "GeometryUtils.hpp"


namespace
{
bool exitDisplayBgr(const cv::Mat& bgr)
{
    cv::imshow("ArucoCubeParser", bgr);
    const auto key = cv::waitKey(1);
    if (key == 27 || key == static_cast<int>('q'))
        return true;
    return false;
}
}

namespace Util
{
ArucoCubeParser::ArucoCubeParser(const std::filesystem::path& replay_video_path, const size_t min_valid_n_markers, const double marker_length, const std::string& marker_res)
    : m_min_valid_n_markers(min_valid_n_markers), m_marker_length(marker_length)
{
    // Initialize capture from either a video file or camera
    if (!replay_video_path.empty()) {
        // Try opening "video.mp4" in the specified folder
        const std::string path = replay_video_path.string();
        m_cap.open(path);
        if (!m_cap.isOpened()) {
            throw std::runtime_error("Failed to open " + path);
        }
        std::cout << "[ArucoCubeParser] Opened video: " << path << std::endl;
    }
    else {
        // Open default camera
        m_cap.open(0);
        if (!m_cap.isOpened()) {
            throw std::runtime_error("Failed to open default camera (device 0).");
        }
        std::cout << "[ArucoCubeParser] Opened live camera (device 0)." << std::endl;
    }
    m_image_size.width = m_cap.get(cv::CAP_PROP_FRAME_WIDTH);
    m_image_size.height = m_cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    // Initialize ArUco Detector
    cv::aruco::PredefinedDictionaryType dict_type{};
    if (marker_res == "4x4")
        dict_type = cv::aruco::DICT_4X4_50;
    else if (marker_res == "5x5")
        dict_type = cv::aruco::DICT_5X5_50;
    else if (marker_res == "6x6")
        dict_type = cv::aruco::DICT_6X6_50;
    else if (marker_res == "7x7")
        dict_type = cv::aruco::DICT_7X7_50;
    else
        throw std::runtime_error("Unknown marker_res type: " + marker_res);

    const auto dictionary = cv::aruco::getPredefinedDictionary(dict_type);
    auto detector_params = cv::aruco::DetectorParameters();
    detector_params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    m_detector = cv::aruco::ArucoDetector(dictionary, detector_params);
}

ArucoCubeParser::~ArucoCubeParser()
{
    if (m_cap.isOpened()) m_cap.release();
}

const cv::Size& ArucoCubeParser::getImageSize() const
{
    return m_image_size;
}

cv::Mat ArucoCubeParser::getBgr(const size_t frame_idx)
{
    return m_bgrs[frame_idx];
}

// processFrames
// Reads all frames until video ends or camera is closed,
// detects markers, checks blur, and stores the results.
std::tuple<Eigen::Matrix3d, Eigen::Vector<double, 5>, std::vector<std::vector<Marker2dOpt>>, std::vector<size_t>>
ArucoCubeParser::processFrames(const std::optional<Eigen::Matrix3d>& camera_matrix_opt, const std::optional<Eigen::Vector<double, 5>>& distortion_coefs_opt)
{
    // Read camera intrinsics, or generate pseudo-params
    Eigen::Matrix3d camera_matrix{
        {static_cast<double>(m_image_size.width), 0., static_cast<double>(m_image_size.width) / 2.},
        {0., static_cast<double>(m_image_size.width), static_cast<double>(m_image_size.height) / 2.},
        {0., 0., 1.} };
    if (camera_matrix_opt.has_value())
        camera_matrix = camera_matrix_opt.value();
    Eigen::Vector<double, 5> distortion_coefs{0., 0., 0., 0., 0.};
    if (distortion_coefs_opt.has_value())
        distortion_coefs = distortion_coefs_opt.value();

    // Loop over frames
    size_t frame_idx = 0;
    while (true) {
        cv::Mat bgr;
        if (!m_cap.read(bgr) || bgr.empty()) {
            // End of video or no camera bgr
            break;
        }

        cv::Mat gray;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

        // Detect ArUco markers
        std::vector<std::vector<cv::Point2f>> markers_image_cv;
        std::vector<int> ids_cv;
        m_detector.detectMarkers(gray, markers_image_cv, ids_cv);

        // Check if there is anything detected
        if (ids_cv.empty()) {
            ++frame_idx;
            if (exitDisplayBgr(bgr)) break;
            continue;
        }

        // Check uniqueness of markers
        std::vector<Marker2d> markers_image_eigen(markers_image_cv.size());
        for (size_t i = 0; i < markers_image_cv.size(); ++i)
            for (size_t j = 0; j < markers_image_cv[i].size(); ++j) {
                markers_image_eigen[i](j, 0) = static_cast<double>(markers_image_cv[i][j].x);
                markers_image_eigen[i](j, 1) = static_cast<double>(markers_image_cv[i][j].y);
            }
        auto [markers_image, ids] = getUniqueMarkers(markers_image_eigen, ids_cv, m_id_limit, 180.0, 0.4);

        // Check if bgr is blurry
        if (isBlurry(gray, markers_image) || ids.size() < m_min_valid_n_markers) {
            ++frame_idx;
            if (exitDisplayBgr(bgr)) break;
            continue;
        }

        // Store
        std::vector<Marker2dOpt> markers_image_ided(m_id_limit);
        for (size_t i = 0; i < ids.size(); ++i) {
            const auto id = ids[i];
            markers_image_ided[static_cast<size_t>(id)] = markers_image[i];
        }
        m_markers_image_ided_framed.push_back(markers_image_ided);
        m_frame_numbers.push_back(frame_idx);
        m_bgrs[frame_idx] = bgr.clone();
        ++frame_idx;

        // Draw
        if (!ids.empty()) {
            ids_cv = ids;
            markers_image_cv.clear();
            for (Eigen::Index i = 0; i < markers_image.size(); ++i) {
                auto& marker_cv = markers_image_cv.emplace_back(std::vector<cv::Point2f>{});
                for (Eigen::Index j = 0; j < markers_image[i].rows(); ++j)
                    marker_cv.push_back({ static_cast<float>(markers_image[i](j, 0)), static_cast<float>(markers_image[i](j, 1)) });
            }
            cv::aruco::drawDetectedMarkers(bgr, markers_image_cv, ids_cv);
        }

        if (exitDisplayBgr(bgr)) break;
    }

    return std::make_tuple(camera_matrix, distortion_coefs, m_markers_image_ided_framed, m_frame_numbers);
}

std::tuple<std::vector<Marker2d>, std::vector<int>> ArucoCubeParser::getUniqueMarkers(
        const std::vector<Marker2d>& markers_image,
        const std::vector<int>& ids,
        const size_t id_limit,
        const double tilt_degree_limit,
        const double xy_ratio_limit)
{
    // For each ID in [0..id_limit), collect indices of markers with that ID.
    std::vector<std::vector<size_t>> indices_per_id(id_limit);
    for (size_t idx = 0; idx < ids.size(); ++idx) {
        const size_t marker_id = ids[idx];
        if (marker_id < id_limit)
            indices_per_id[marker_id].push_back(idx);
    }

    // Reject markers whose 0->1 corner tilt is too large,
    // or whose bounding box ratio is too small
    // Keep the valid markers grouped by ID
    std::vector<std::vector<Marker2d>> sorted_markers_image(id_limit);
    for (size_t unique_id = 0; unique_id < id_limit; ++unique_id) {
        const auto& idx_list = indices_per_id[unique_id];
        if (idx_list.empty()) continue;

        for (const auto idx : idx_list) {
            const Marker2d& marker_image = markers_image[idx];
            // Compute 0->1 side orientation in (x,y) space
            // corners2d.row(1) - corners2d.row(0)
            const Eigen::RowVector2d corner0 = marker_image.row(0);
            const Eigen::RowVector2d corner1 = marker_image.row(1);
            const Eigen::RowVector2d diff = (corner1 - corner0).cwiseAbs();  // coefficient-wise absolute difference
            const double norm_diff = diff.norm();
            // Guard against degenerate zero-length
            if (norm_diff < 1e-20) continue;
            // angle in degrees between x-axis and the 0->1 edge
            const double degree = std::acos(diff(0) / norm_diff) * 180.0 / CV_PI;

            // Compute bounding box ratio in (x,y)
            // ratio = (min(width, height) / max(width, height))
            const Eigen::Vector2d min_corner = marker_image.colwise().minCoeff();
            const Eigen::Vector2d max_corner = marker_image.colwise().maxCoeff();
            const Eigen::Vector2d tl_to_rb = max_corner - min_corner;  // (width, height)
            const double minor_len = std::min(tl_to_rb(0), tl_to_rb(1));
            const double major_len = std::max(tl_to_rb(0), tl_to_rb(1));
            // Avoid division by zero if bounding box is degenerate
            if (major_len < 1e-20) continue;
            double ratio = minor_len / major_len;

            // Check thresholds, keep corners if valid.
            if (degree < tilt_degree_limit && ratio > xy_ratio_limit) {
                sorted_markers_image[unique_id].push_back(marker_image);
            }
        }
    }

    // For each ID, pick the marker with largest area.
    // We'll use the shoelace formula in (x,y)
    std::vector<Marker2d> unique_markers_image;
    std::vector<int> unique_ids;
    for (size_t unique_id = 0; unique_id < id_limit; ++unique_id) {
        const auto& duplicate_markers_image = sorted_markers_image[unique_id];
        if (duplicate_markers_image.empty()) continue;

        double max_area = 0.0;
        size_t best_idx = 0;
        for (size_t i = 0; i < duplicate_markers_image.size(); ++i)
        {
            // Again, extract (x,y) only:
            const Marker2d& marker_image = duplicate_markers_image[i];

            // Shoelace formula
            double area = 0.0;
            for (Eigen::Index k = 0; k < 4; ++k)
            {
                Eigen::Index k_next = (k + 1) % 4;
                area += marker_image(k, 0) * marker_image(k_next, 1) - marker_image(k, 1) * marker_image(k_next, 0);
            }
            area = 0.5 * std::fabs(area);

            if (area > max_area)
            {
                max_area = area;
                best_idx = i;
            }
        }

        // Keep the largest-area marker for this ID.
        unique_markers_image.push_back(duplicate_markers_image[best_idx]);
        unique_ids.push_back(static_cast<int>(unique_id));
    }

    return std::make_tuple(unique_markers_image, unique_ids);
}

// isBlurry
// For each marker's bounding box, compute Laplacian variance in that ROI.
// If average < blur_threshold, the frame is considered blurry.
bool ArucoCubeParser::isBlurry(const cv::Mat& gray, const std::vector<Marker2d>& markers_image) const
{
    // If no markers, treat as blurry (or decide otherwise).
    if (markers_image.empty()) {
        return true;
    }

    std::vector<double> variances;
    variances.reserve(markers_image.size());

    for (const auto& marker_image : markers_image)
    {
        // Compute bounding box using minCoeff() and maxCoeff().
        const double minX = marker_image.col(0).minCoeff();
        const double maxX = marker_image.col(0).maxCoeff();
        const double minY = marker_image.col(1).minCoeff();
        const double maxY = marker_image.col(1).maxCoeff();

        // Convert to integer bounding box within image bounds (floor for min, ceil for max).
        const int x1 = std::max(0, static_cast<int>(std::floor(minX)));
        const int y1 = std::max(0, static_cast<int>(std::floor(minY)));
        const int x2 = std::min(gray.cols - 1, static_cast<int>(std::ceil(maxX)));
        const int y2 = std::min(gray.rows - 1, static_cast<int>(std::ceil(maxY)));

        if (x1 > x2 || y1 > y2) {
            // Invalid bounding box
            continue;
        }

        // Extract ROI from the grayscale image.
        const cv::Rect roi_rect(
            x1,
            y1,
            (x2 - x1 + 1),
            (y2 - y1 + 1)
        );
        const cv::Mat roi = gray(roi_rect);

        // Compute the Laplacian and then the variance of the Laplacian.
        cv::Mat lap;
        cv::Laplacian(roi, lap, CV_64F);

        cv::Scalar mean_val, stdev_val;
        cv::meanStdDev(lap, mean_val, stdev_val);

        const double variance = stdev_val[0] * stdev_val[0];
        variances.push_back(variance);
    }

    // If no valid bounding boxes, consider it blurry (or decide otherwise).
    if (variances.empty()) {
        return true;
    }

    // Compute the mean of all variances
    const double sum_v = std::accumulate(variances.begin(), variances.end(), 0.0);
    const double meanV = sum_v / static_cast<double>(variances.size());

    // Compare against your blur threshold
    return (meanV < m_blur_threshold);
}
}
