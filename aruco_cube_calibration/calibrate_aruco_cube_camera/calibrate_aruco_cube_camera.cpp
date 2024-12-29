/***********************************************************
 * Example: ArucoCubeCalibratorWithDlib.cpp
 *
 * Requires:
 *   - dlib
 *   - OpenCV
 *   - Proper linkage to both
 *
 * Conventions:
 *   - Function names: lowerCamelCase (e.g. someFunctionName)
 *   - Variable names: lower_snake_case (e.g. some_variable_name)
 ***********************************************************/

#include <iostream>
#include <fstream>
#include <limits>
#include <map>
#include <algorithm>

#include <cxxopts.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <dlib/optimization.h>
#include <dlib/matrix.h>

#include "utils/Types.hpp"
#include "utils/ArucoCubeParser.hpp"
#include "utils/GeometryUtils.hpp"
#include "utils/PlotUtils.hpp"
#include "utils/Json.hpp"


// Conversion functions for OpenCV operations
std::vector<cv::Point3d> eigenToCV(const Eigen::MatrixX3d& pts) {
    std::vector<cv::Point3d> ret(pts.rows());
    for (size_t i = 0; i < pts.rows(); ++i) {
        ret[i] = cv::Point3d(pts(i, 0), pts(i, 1), pts(i, 2));
    }
    return ret;
}
Eigen::MatrixX2d cvToEigen(const std::vector<cv::Point2d>& pts) {
    Eigen::MatrixX2d ret;
    ret.resize(pts.size(), Eigen::NoChange);
    for (size_t i = 0; i < pts.size(); ++i) {
        ret.row(i) = Eigen::RowVector2d(pts[i].x, pts[i].y);
    }
    return ret;
}

// Initial corners on cube with marker length
auto getInitialCornersInCube(const double marker_length)
{
    std::array<Marker3d, 6> corners_cube_6x4x3;
    const Marker3d corners_local{
        { marker_length / 2, -marker_length / 2, 0},
        {-marker_length / 2, -marker_length / 2, 0},
        {-marker_length / 2,  marker_length / 2, 0},
        { marker_length / 2,  marker_length / 2, 0}
    };
    const double half_size = (marker_length + marker_length / 3.0) * 0.5;
    const std::vector<Eigen::RowVector3d> face_origins_cube{
        Eigen::RowVector3d(0,           0,           half_size),
        Eigen::RowVector3d(0,          -half_size,   0),
        Eigen::RowVector3d(half_size,   0,           0),
        Eigen::RowVector3d(0,           0,          -half_size),
        Eigen::RowVector3d(-half_size,  0,           0),
        Eigen::RowVector3d(0,           half_size,   0)
    };
    corners_cube_6x4x3[0] = corners_local.rowwise() + face_origins_cube[0];                                   // front
    corners_cube_6x4x3[1] = Util::rotateAboutX(corners_local, CV_PI / 2).rowwise() + face_origins_cube[1];    // top
    corners_cube_6x4x3[2] = Util::rotateAboutY(corners_local, CV_PI / 2).rowwise() + face_origins_cube[2];    // left
    corners_cube_6x4x3[3] = Util::rotateAboutY(corners_local, CV_PI).rowwise() + face_origins_cube[3];        // back
    corners_cube_6x4x3[4] = Util::rotateAboutY(corners_local, -CV_PI / 2).rowwise() + face_origins_cube[4];   // right
    corners_cube_6x4x3[5] = Util::rotateAboutX(corners_local, -CV_PI / 2).rowwise() + face_origins_cube[5];   // bottom
    return corners_cube_6x4x3;
}

// Params lower/upper bounds
auto getBounds(const Eigen::Index params_count, const Eigen::Matrix3d& camera_matrix, const Eigen::Vector<double, 5>& distortion_coefs)
{
    Eigen::VectorXd lower_bounds, upper_bounds;
    lower_bounds.resize(params_count);
    upper_bounds.resize(params_count);
    lower_bounds.fill(-1e+1);
    upper_bounds.fill(1e+1);

    const double fx = camera_matrix(0, 0);
    const double cx = camera_matrix(0, 2);
    const double fy = camera_matrix(1, 1);
    const double cy = camera_matrix(1, 2);
    // camera_matrix
    lower_bounds(0) = 0.5 * fx;   // fx
    lower_bounds(1) = -0.1;       // skew
    lower_bounds(2) = 0.5 * cx;   // cx
    lower_bounds(3) = 0.5 * fy;   // fy
    lower_bounds(4) = 0.5 * cy;   // cy
    upper_bounds(0) = 1.5 * fx;
    upper_bounds(1) = 0.1;
    upper_bounds(2) = 1.5 * cx;
    upper_bounds(3) = 1.5 * fy;
    upper_bounds(4) = 1.5 * cy;

    // distortion_coefs
    lower_bounds(5) = -0.5; // k1
    lower_bounds(6) = -0.5; // k2
    lower_bounds(7) = -0.1; // p1
    lower_bounds(8) = -0.1; // p2
    lower_bounds(9) = -0.5; // k3
    upper_bounds(5) = 0.5;
    upper_bounds(6) = 0.5;
    upper_bounds(7) = 0.1;
    upper_bounds(8) = 0.1;
    upper_bounds(9) = 0.5;
    return std::make_tuple(lower_bounds, upper_bounds);
}

// Un/Pack: camera intrinsics(5) + distCoeffs(5) + corners(6*4*3=72) + poses(n_frames*6)
Eigen::VectorXd packParams(
    const Eigen::Matrix3d& camera_matrix,
    const Eigen::Vector<double, 5>& distortion_coefs,
    const std::array<Marker3d, 6>& corners_cube_6x4x3,
    const MatrixX6d& cube_to_camera_rtvecs_nx6)
{
    const auto total_size = 5 + 5 + 6 * 4 * 3 + cube_to_camera_rtvecs_nx6.size();
    Eigen::VectorXd params(total_size);
    double* p_data = params.data();

    // camera intrinsics
    p_data[0] = camera_matrix(0, 0); // fx
    p_data[1] = camera_matrix(0, 1); // skew
    p_data[2] = camera_matrix(0, 2); // cx
    p_data[3] = camera_matrix(1, 1); // fy
    p_data[4] = camera_matrix(1, 2); // cy
    std::copy(distortion_coefs.begin(), distortion_coefs.end(), p_data + 5);

    // corners (6 faces * 4 corners * 3 coords)
    size_t p_idx = 10;
    for (size_t face_idx = 0; face_idx < 6; ++face_idx) {
        auto& corners_cube_4x3 = corners_cube_6x4x3[face_idx];
        std::copy(corners_cube_4x3.data(), corners_cube_4x3.data() + 4 * 3, p_data + p_idx);
        p_idx += 4 * 3;
    }

    // poses
    std::copy(cube_to_camera_rtvecs_nx6.data(), cube_to_camera_rtvecs_nx6.data() + cube_to_camera_rtvecs_nx6.size(), p_data + p_idx);
    return params;
}
auto unpackParams(const Eigen::VectorXd& params)
{
    double* p_data = const_cast<double*>(params.data());

    // camera intrinsics
    Eigen::Matrix3d camera_matrix = Eigen::Matrix3d::Identity();
    camera_matrix(0, 0) = p_data[0]; // fx
    camera_matrix(0, 1) = p_data[1]; // skew
    camera_matrix(0, 2) = p_data[2]; // cx
    camera_matrix(1, 1) = p_data[3]; // fy
    camera_matrix(1, 2) = p_data[4]; // cy

    Eigen::Vector<double, 5> distortion_coefs;
    std::copy(p_data + 5, p_data + 10, distortion_coefs.begin());

    // corners (6 faces * 4 corners * 3 coords)
    size_t p_idx = 10;
    std::array<Marker3d, 6> corners_cube_6x4x3;
    for (size_t face_idx = 0; face_idx < 6; ++face_idx) {
        std::copy(p_data + p_idx, p_data + p_idx + 4 * 3, corners_cube_6x4x3[face_idx].data());
        p_idx += 4 * 3;
    }

    // poses
    MatrixX6d cube_to_camera_rtvecs_nx6;
    cube_to_camera_rtvecs_nx6.resize((params.size() - p_idx) / 6, Eigen::NoChange);
    std::copy(p_data + p_idx, p_data + p_idx + cube_to_camera_rtvecs_nx6.size(), cube_to_camera_rtvecs_nx6.data());
    return std::make_tuple(camera_matrix, distortion_coefs, corners_cube_6x4x3, cube_to_camera_rtvecs_nx6);
}

// Estimate initial poses
MatrixX6d estimateCubePoses(
    const std::vector<std::vector<std::pair<Marker2d, size_t>>>& markers_image_id_pair_framed,
    const std::array<Marker3d, 6>& corners_cube_6x4x3,
    const Eigen::Matrix3d& camera_matrix,
    const Eigen::Vector<double, 5>& distortion_coefs)
{
    MatrixX6d ret;
    for (const auto& markers_image_id_pair : markers_image_id_pair_framed) {
        Eigen::MatrixX2d corners_image_all;
        Eigen::MatrixX3d corners_cube_all;
        for (const auto& marker_image_id_pair : markers_image_id_pair) {
            const auto face_id = marker_image_id_pair.second;
            const auto& markers_image = marker_image_id_pair.first;
            const auto& corners_cube = corners_cube_6x4x3[face_id];
            corners_image_all.conservativeResize(corners_image_all.rows() + markers_image.rows(), Eigen::NoChange);
            corners_cube_all.conservativeResize(corners_cube_all.rows() + corners_cube.rows(), Eigen::NoChange);
            corners_image_all.bottomRows(4) = markers_image;
            corners_cube_all.bottomRows(4) = corners_cube;

        }
        const auto [rvec, tvec] = Util::getPointcloudPose(corners_cube_all, corners_image_all, camera_matrix, distortion_coefs);
        ret.conservativeResize(ret.rows() + 1, Eigen::NoChange);
        ret.bottomRows(1) = Eigen::RowVector<double, 6>{ rvec(0), rvec(1), rvec(2), tvec(0), tvec(1), tvec(2) };
    }
    return ret;
}

// Return a single-valued cost
auto getCostKeyPairs(const std::vector<std::vector<std::pair<Marker2d, size_t>>>& markers_image_id_pair_framed)
{
    std::map<std::pair<size_t, size_t>, double> cost_key_pairs;
    for (size_t frame_idx = 0; frame_idx < markers_image_id_pair_framed.size(); ++frame_idx)
        for (const auto& marker_image : markers_image_id_pair_framed[frame_idx])
            cost_key_pairs[std::make_pair(frame_idx, marker_image.second)] = 0;
    return cost_key_pairs;
}
double costCubePose(
    const Eigen::VectorXd& params,
    const std::vector<std::vector<std::pair<Marker2d, size_t>>>& markers_image_id_pair_framed,
    std::map<std::pair<size_t, size_t>, double> cost_key_pairs = {})
{
    const auto [camera_matrix, distortion_coefs, corners_cube_6x4x3, cube_to_camera_rtvecs_nx6] = unpackParams(params);
    const auto [camera_matrix_cv, distortion_coefs_cv] = Util::toCvCameraIntrinsics(camera_matrix, distortion_coefs);
    if (cost_key_pairs.empty())
        cost_key_pairs = getCostKeyPairs(markers_image_id_pair_framed);

    Util::forEachIdx<size_t>(markers_image_id_pair_framed.size(), [&](const size_t frame_idx) {
        const Eigen::Matrix4d cube_to_camera = Util::rtToMat4x4(cube_to_camera_rtvecs_nx6.row(frame_idx));

        for (const auto& markers_image_id_pair : markers_image_id_pair_framed[frame_idx]) {
            const auto& corners_image = markers_image_id_pair.first;
            const auto face_id = markers_image_id_pair.second;
            const auto corners_camera_prime = Util::transform4x4(cube_to_camera, corners_cube_6x4x3[face_id]);

            std::vector<cv::Point2d> corners_image_prime_cv;
            cv::projectPoints(
                eigenToCV(corners_camera_prime), cv::Mat::zeros(3, 1, CV_64F), cv::Mat::zeros(3, 1, CV_64F),
                camera_matrix_cv, distortion_coefs_cv, corners_image_prime_cv);

            const auto corners_image_prime = cvToEigen(corners_image_prime_cv);
            const Eigen::MatrixX2d error = (corners_image - corners_image_prime).cwiseAbs2();
            const auto key = std::make_pair(frame_idx, face_id);
            cost_key_pairs[key] = error.sum();
        }
    });

    const double cost_sum = std::accumulate(cost_key_pairs.begin(), cost_key_pairs.end(), 0.0,
        [](const double sum, const std::pair<std::pair<size_t, size_t>, double>& cost_key_pair) {
            return sum + cost_key_pair.second;
        });
    const double cost = cost_sum;
    return cost;
}

Eigen::VectorXd optimizeParams(
    const Eigen::VectorXd& lower_bounds,
    const Eigen::VectorXd& upper_bounds,
    const Eigen::VectorXd& initial_params,
    const std::vector<std::vector<std::pair<Marker2d, size_t>>>& markers_image_id_pair_framed)
{
    // Convert initial parameters to dlib matrix
    dlib::matrix<double, 0, 1> dlib_params(initial_params.size());
    for (Eigen::Index i = 0; i < initial_params.size(); ++i)
        dlib_params(i) = initial_params(i);

    // Cost-key pairs
    std::map<std::pair<size_t, size_t>, double> cost_key_pairs = getCostKeyPairs(markers_image_id_pair_framed);

    // Define cost function using lambda
    auto cost_function = [&](const dlib::matrix<double, 0, 1>& params) {
        // Convert dlib matrix to Eigen vector
        Eigen::VectorXd eigen_params(params.size());
        std::copy(params.begin(), params.end(), eigen_params.begin());
        return costCubePose(eigen_params, markers_image_id_pair_framed, cost_key_pairs);
    };

    try {
        dlib::matrix<double> lower_bounds_dlib(lower_bounds.rows(), lower_bounds.cols());
        dlib::matrix<double> upper_bounds_dlib(upper_bounds.rows(), upper_bounds.cols());
        std::copy(lower_bounds.begin(), lower_bounds.end(), lower_bounds_dlib.begin());
        std::copy(upper_bounds.begin(), upper_bounds.end(), upper_bounds_dlib.begin());

        const double final_cost = dlib::find_min_box_constrained(dlib::bfgs_search_strategy(),
            dlib::objective_delta_stop_strategy(1e-7).be_verbose(),
            cost_function, dlib::derivative(cost_function), dlib_params,
            lower_bounds_dlib,
            upper_bounds_dlib);

        // Convert result back to Eigen vector
        Eigen::VectorXd optimized_params(dlib_params.size());
        std::copy(dlib_params.begin(), dlib_params.end(), optimized_params.begin());
        return optimized_params;
    }
    catch (const std::exception& e) {
        std::cerr << "Optimization failed: " << e.what() << std::endl;
        return initial_params; // Return initial parameters if optimization fails
    }
}

auto getMarkerIdPairsFramed(const std::vector<std::vector<Marker2dOpt>>& markers_image_ided_framed)
{
    std::vector<std::vector<std::pair<Marker2d, size_t>>> markers_image_id_pair_framed;
    for (size_t frame_idx = 0; frame_idx < markers_image_ided_framed.size(); ++frame_idx) {
        auto& markers_image_id_pair = markers_image_id_pair_framed.emplace_back(std::vector<std::pair<Marker2d, size_t>>{});
        for (size_t face_idx = 0; face_idx < markers_image_ided_framed[frame_idx].size(); ++face_idx) {
            if (!markers_image_ided_framed[frame_idx][face_idx].has_value()) continue;
            markers_image_id_pair.push_back({ markers_image_ided_framed[frame_idx][face_idx].value(), face_idx });
        }
    }
    return markers_image_id_pair_framed;
}

auto getUnbiasedIdedMarkers(const std::vector<std::vector<Marker2dOpt>>& markers_image_ided_framed, const std::vector<size_t> frame_numbers, const size_t n_samples) {
    // Group frames by visible 3-face combos
    std::map<std::vector<size_t>, std::vector<std::pair<size_t, Eigen::Vector2d>>> face_combo_groups;
    for (size_t idx = 0; idx < markers_image_ided_framed.size(); ++idx) {
        std::vector<size_t> visible_faces;
        for (size_t face_id = 0; face_id < markers_image_ided_framed[idx].size(); ++face_id) {
            if (!markers_image_ided_framed[idx][face_id].has_value())
                continue;
            visible_faces.push_back(face_id);
        }

        if (visible_faces.size() != 3)
            continue;

        std::sort(visible_faces.begin(), visible_faces.end());
        Eigen::MatrixX2d all_pts;
        for (const auto face_id : visible_faces) {
            const auto& marker_image = markers_image_ided_framed[idx][face_id].value();
            all_pts.conservativeResize(all_pts.rows() + marker_image.rows(), Eigen::NoChange);
            all_pts.bottomRows(marker_image.rows()) = marker_image;
        }

        const Eigen::Vector2d mean_pt = all_pts.colwise().mean().transpose();
        face_combo_groups[visible_faces].push_back(std::make_pair(idx, mean_pt));
    };
    std::cout << "[INFO] Found " << face_combo_groups.size() << " valid face combos.\n";

    const size_t samples_per_combo = n_samples / face_combo_groups.size();
    std::vector<size_t> insample_indices;
    for (const auto& face_combo_group : face_combo_groups) {
        const auto& frame_means = face_combo_group.second; // (frame_idx, mean_pt)
        std::vector<size_t> frame_idc;
        std::vector<Eigen::Vector2d> means;
        for (const auto& frame_mean : frame_means) {
            frame_idc.push_back(frame_mean.first);
            means.push_back(frame_mean.second);
        }
        // get spatially distributed frames
        const std::vector<size_t> selected = Util::getSpatiallyDistributedFrameIndices(means, samples_per_combo);
        for (const auto s : selected) {
            insample_indices.push_back(frame_idc[s]);
        }
    }

    // Filter
    std::vector<std::vector<Marker2dOpt>> markers_image_ided_framed_unbiased;
    markers_image_ided_framed_unbiased.reserve(insample_indices.size());
    std::vector<size_t> frame_numbers_unbiased;
    frame_numbers_unbiased.reserve(insample_indices.size());
    for (auto idx : insample_indices) {
        markers_image_ided_framed_unbiased.push_back(markers_image_ided_framed[idx]);
        frame_numbers_unbiased.push_back(frame_numbers[idx]);
    }
    return std::make_tuple(getMarkerIdPairsFramed(markers_image_ided_framed_unbiased), frame_numbers_unbiased);
}

int main(int argc, char** argv) try {
    // Parse args
    cxxopts::Options options("calibrate_aruco_cube_camera", "Calibrate ArUco cube camera parameters");
    options.add_options()
        ("nsamples", "Number of samples", cxxopts::value<int>())
        ("marker-length", "Marker length", cxxopts::value<double>())
        ("cam-json", "Camera intrinsics json", cxxopts::value<std::string>()->default_value(""))
        ("replay", "Video replay folder", cxxopts::value<std::string>()->default_value(""))
        ("h,help", "Print usage");
    const auto parse_result = options.parse(argc, argv);
    if (parse_result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }
    if (!parse_result.count("nsamples") || !parse_result.count("marker-length")) {
        std::cerr << "Missing required arguments\n" << options.help() << std::endl;
        return -1;
    }

    const auto n_samples = parse_result["nsamples"].as<int>();
    const auto marker_length = parse_result["marker-length"].as<double>();
    const auto replay_path = parse_result["replay"].as<std::string>();
    std::cout << "[INFO] n_samples = " << n_samples << ", marker_length = " << marker_length << std::endl;
    if (n_samples <= 0 || marker_length <= 0.0) {
        std::cerr << "Invalid arguments: n_samples and marker_length must be positive\n";
        return 1;
    }

    // Parse calibration json if supplied
    auto [corners_cube_6x4x3_opt, camera_matrix_opt, distortion_coefs_opt] = Util::readArucoCubeJson(parse_result["cam-json"].as<std::string>());
    // Parse all the samples available from image stream
    Util::ArucoCubeParser aruco_cube_parser(replay_path, 3, marker_length);
    auto [camera_matrix, distortion_coefs, markers_image_ided_framed_all, frame_numbers_all] = aruco_cube_parser.processFrames(camera_matrix_opt, distortion_coefs_opt);
    // Get Unbiased samples
    auto [markers_image_id_pair_framed, frame_numbers] = getUnbiasedIdedMarkers(markers_image_ided_framed_all, frame_numbers_all, static_cast<size_t>(n_samples));
    const double corners_count = std::accumulate(markers_image_id_pair_framed.begin(), markers_image_id_pair_framed.end(), 0.0,
        [](const double corner_count_sum, const std::vector<std::pair<Marker2d, size_t>>& markers_image_id_pair) {
            return corner_count_sum + static_cast<double>(markers_image_id_pair.size()) * 4.;
        });

    // Build initial corners
    const std::array<Marker3d, 6> corners_cube_6x4x3 = getInitialCornersInCube(marker_length);
    // Estimate initial poses
    MatrixX6d cube_to_camera_rtvecs_nx6 = estimateCubePoses(markers_image_id_pair_framed, corners_cube_6x4x3, camera_matrix, distortion_coefs);

    const Eigen::VectorXd initial_params = packParams(camera_matrix, distortion_coefs, corners_cube_6x4x3, cube_to_camera_rtvecs_nx6);
    const auto [lower_bounds, upper_bounds] = getBounds(initial_params.size(), camera_matrix, distortion_coefs);

    const double initial_cost = costCubePose(initial_params, markers_image_id_pair_framed) / corners_count;
    std::cout << "[INFO] Initial cost: " << initial_cost << std::endl;

    // Solve with dlib
    const Eigen::VectorXd result_params = optimizeParams(lower_bounds, upper_bounds, initial_params, markers_image_id_pair_framed);
    const double final_cost = costCubePose(result_params, markers_image_id_pair_framed) / corners_count;
    std::cout << "[INFO] Final cost: " << final_cost << std::endl;

    // Unpack
    auto [camera_matrix_prime, distortion_coefs_prime, corners_cube_6x4x3_prime, cube_to_camera_rtvecs_nx6_prime] = unpackParams(result_params);
    const auto [camera_matrix_prime_cv, distortion_coefs_prime_cv] = Util::toCvCameraIntrinsics(camera_matrix_prime, distortion_coefs_prime);

    const auto image_size = aruco_cube_parser.getImageSize();
    const auto camera_matrix_corrected_cv = cv::getOptimalNewCameraMatrix(camera_matrix_prime_cv, distortion_coefs_prime_cv, image_size, 1.0, image_size, {});
    const auto [camera_matrix_corrected, distortion_coefs_corrected] = Util::toEigenCameraIntrinsics(camera_matrix_corrected_cv, cv::Mat::zeros(1, 5, CV_64FC1));

    std::cout << "[INFO] initial camera_matrix:\n" << camera_matrix << std::endl;
    std::cout << "[INFO] optimized camera_matrix:\n" << camera_matrix_prime << std::endl;
    std::cout << "[INFO] initial dist_coeffs:\n" << distortion_coefs << std::endl;
    std::cout << "[INFO] optimized dist_coeffs:\n" << distortion_coefs_prime << std::endl;

    corners_cube_6x4x3_prime = Util::alignCornersToOrthogonalFrame(corners_cube_6x4x3_prime);

    // Write JSON
    const auto replay_dir = std::filesystem::path(replay_path).parent_path();
    Util::writeArucoCubeJson(replay_dir, corners_cube_6x4x3_prime, camera_matrix_prime, distortion_coefs_prime);

    markers_image_id_pair_framed = getMarkerIdPairsFramed(markers_image_ided_framed_all);
    frame_numbers = frame_numbers_all;
    // Visualization: re-use markers_image_id_pair_framed with some faces reduced
    std::vector<std::vector<std::pair<Marker2d, size_t>>> markers_image_id_pair_framed_reduced;
    for(auto& markers_image_id_pair : markers_image_id_pair_framed) {
        auto& markers_image_id_pair_reduced = markers_image_id_pair_framed_reduced.emplace_back(std::vector<std::pair<Marker2d, size_t>>{});
        for (auto& marker_image_id_pair : markers_image_id_pair) {
            if (marker_image_id_pair.second == 0 || marker_image_id_pair.second == 3)
                markers_image_id_pair_reduced.push_back(marker_image_id_pair);
        }
    }
    cube_to_camera_rtvecs_nx6 = estimateCubePoses(markers_image_id_pair_framed, corners_cube_6x4x3, camera_matrix, distortion_coefs);
    cube_to_camera_rtvecs_nx6_prime = estimateCubePoses(markers_image_id_pair_framed_reduced, corners_cube_6x4x3_prime, camera_matrix_prime, distortion_coefs_prime);

    // Save a video example
    const std::string out_video_path = (replay_dir / "initial_to_result.avi").string();
    cv::VideoWriter writer;
    writer.open(out_video_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, image_size, true);
    if(!writer.isOpened()) {
        std::cerr << "Cannot open video file for writing.\n";
        return -1;
    }

    const auto [camera_matrix_cv, distortion_coefs_cv] = Util::toCvCameraIntrinsics(camera_matrix, distortion_coefs);
    for (size_t idx = 0; idx < frame_numbers.size(); ++idx) {
        const auto frame_idx = frame_numbers[idx];
        cv::Mat bgr = aruco_cube_parser.getBgr(frame_idx);

        // Draw red diamond as optimized corners (used for pnp-estimation)
        for(const auto& marker_image_id_pair_reduced : markers_image_id_pair_framed_reduced[idx]) {
            const auto& marker_image = marker_image_id_pair_reduced.first;
            for(Eigen::Index r = 0; r < marker_image.rows(); ++r)
                cv::drawMarker(bgr, cv::Point2f{ static_cast<float>(marker_image(r, 0)), static_cast<float>(marker_image(r, 1)) }, cv::Scalar(0, 0, 255), cv::MARKER_DIAMOND, 20, 1);
        }

        // Collect face ids
        std::vector<int> ids;
        for(const auto& marker_image_id_pair : markers_image_id_pair_framed[idx])
            ids.push_back(static_cast<int>(marker_image_id_pair.second));

        // Draw yellow cross as pre-optimized corners and green cross as post-optimized corners
        std::vector<Marker3d> pre_optimized_corners_cube;
        std::vector<Marker3d> post_optimized_corners_cube;
        for (const auto id : ids) {
            pre_optimized_corners_cube.push_back(corners_cube_6x4x3[id]);
            post_optimized_corners_cube.push_back(corners_cube_6x4x3_prime[id]);
        }
        Util::drawProjectedCorners(bgr, cube_to_camera_rtvecs_nx6.row(idx), pre_optimized_corners_cube,
            camera_matrix_cv, distortion_coefs_cv, cv::Scalar(0, 255, 255), cv::MARKER_TILTED_CROSS);
        Util::drawProjectedCorners(bgr, cube_to_camera_rtvecs_nx6_prime.row(idx), post_optimized_corners_cube,
            camera_matrix_prime_cv, distortion_coefs_prime_cv, cv::Scalar(0, 255, 0), cv::MARKER_CROSS);
        Util::drawMarkerLines(bgr, cube_to_camera_rtvecs_nx6_prime.row(idx), post_optimized_corners_cube,
            camera_matrix_prime_cv, distortion_coefs_prime_cv, cv::Scalar(0, 255, 0));

        // draw axes
        const Eigen::RowVector<double, 6> cube_to_camera_rtvecs_prime = cube_to_camera_rtvecs_nx6_prime.row(idx);
        const cv::Mat rvec = (cv::Mat_<double>(3, 1) << cube_to_camera_rtvecs_prime(0), cube_to_camera_rtvecs_prime(1), cube_to_camera_rtvecs_prime(2));
        const cv::Mat tvec = (cv::Mat_<double>(3, 1) << cube_to_camera_rtvecs_prime(3), cube_to_camera_rtvecs_prime(4), cube_to_camera_rtvecs_prime(5));
        cv::drawFrameAxes(bgr, camera_matrix_prime_cv, distortion_coefs_prime_cv, rvec, tvec, marker_length);

        if(writer.isOpened())
            writer.write(bgr);
    }
    writer.release();
    std::cout << "[INFO] Done.\n";

    return 0;
}
catch (const std::runtime_error& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
}
