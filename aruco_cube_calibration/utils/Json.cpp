#include "Json.hpp"

#include <fstream>
#include <iostream>
#include <filesystem>
#include <nlohmann/json.hpp>


namespace Util
{
template<typename Derived>
nlohmann::json eigenToJson(const Eigen::MatrixBase<Derived>& mat)
{
    nlohmann::json json_mat = nlohmann::json::array();
    for (size_t r = 0; r < mat.rows(); ++r) {
        nlohmann::json row = nlohmann::json::array();
        for (size_t c = 0; c < mat.cols(); ++c)
            row.push_back(mat(r, c));
        json_mat.push_back(row);
    }
    return json_mat;
}

template <typename MatType>
MatType jsonToEigen(const nlohmann::json& j)
{
    MatType mat;
    for (size_t r = 0; r < mat.rows(); ++r)
        for (size_t c = 0; c < mat.cols(); ++c)
            mat(r, c) = j.at(r).at(c).get<double>();
    return mat;
}

void writeArucoCubeJson(
    const std::filesystem::path& replay_folder,
    const std::array<Marker3d, 6>& corners_cube_6x4x3,
    const Eigen::Matrix3d& camera_matrix,
    const Eigen::Vector<double, 5>& distortion_coefs)
{
    // Build a JSON object
    nlohmann::json aruco_cube;
    nlohmann::json markers_json = nlohmann::json::array();
    for (size_t marker_idx = 0; marker_idx < 6; ++marker_idx)
        markers_json.push_back(eigenToJson(corners_cube_6x4x3[marker_idx]));
    aruco_cube["markers_in_cube"] = markers_json;
    aruco_cube["camera_matrix"] = eigenToJson(camera_matrix);
    aruco_cube["distortion_coefs"] = eigenToJson(distortion_coefs.transpose());

    // Create the output folder if needed
    std::filesystem::path outputDir = replay_folder / "aruco_cube";
    std::filesystem::create_directories(outputDir);

    // Timestamp-based filename, e.g. "20241227-151230-aruco_cube.json"
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d-%H%M%S") << "-aruco_cube.json";
    std::string filename = (outputDir / oss.str()).string();

    // Write JSON to file
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening " << filename << "\n";
        return;
    }
    file << std::setw(4) << aruco_cube << std::endl;
    std::cout << "Saved JSON to " << filename << std::endl;
}

std::tuple<std::optional<std::array<Marker3d, 6>>, std::optional<Eigen::Matrix3d>, std::optional<Eigen::Vector<double, 5>>>
readArucoCubeJson(const std::string& jsonFilePath)
{
    if (!std::filesystem::exists(jsonFilePath)) {
        std::cout << "JSON file does not exist: " + jsonFilePath << std::endl;
        return {};
    }

    std::ifstream file(jsonFilePath);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file: " + jsonFilePath);
    }
    nlohmann::json aruco_cube;
    file >> aruco_cube;
    file.close();

    std::optional<std::array<Marker3d, 6>> corners_cube_6x4x3 = {};
    if (aruco_cube.contains("markers_in_cube")) {
        const auto& markers_json = aruco_cube.at("markers_in_cube");
        corners_cube_6x4x3 = std::array<Marker3d, 6>{};
        for (size_t marker_idx = 0; marker_idx < markers_json.size(); ++marker_idx) {
            // Each marker is expected to be a 4x3 matrix in JSON:
            // [
            //   [ x00, x01, x02 ],
            //   [ x10, x11, x12 ],
            //   [ x20, x21, x22 ],
            //   [ x30, x31, x32 ]
            // ]
            // Parsing it into an Eigen::Matrix<double,4,3>
            corners_cube_6x4x3.value()[marker_idx] = jsonToEigen<Marker3d>(markers_json.at(marker_idx));
        }
    }

    // Camera matrix
    std::optional<Eigen::Matrix3d> camera_matrix = {};
    if (aruco_cube.contains("camera_matrix")) {
        camera_matrix = jsonToEigen<Eigen::Matrix3d>(aruco_cube.at("camera_matrix"));
    }

    // Distortion coefficients
    std::optional<Eigen::Vector<double, 5>> distortion_coefs = {};
    if (aruco_cube.contains("distortion_coefs")) {
        distortion_coefs = jsonToEigen<Eigen::RowVector<double, 5>>(aruco_cube.at("distortion_coefs")).transpose();
    }

    return std::make_tuple(corners_cube_6x4x3, camera_matrix, distortion_coefs);
}
}
