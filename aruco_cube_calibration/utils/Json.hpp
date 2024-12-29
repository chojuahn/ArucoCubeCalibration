#pragma once
#include <filesystem>

#include <Eigen/Core>

#include "utils/Types.hpp"

namespace Util
{
void writeArucoCubeJson(
    const std::filesystem::path& replay_folder,
    const std::array<Marker3d, 6>& corners_cube_6x4x3,
    const Eigen::Matrix3d& camera_matrix,
    const Eigen::Vector<double, 5>& dist_coeffs);

std::tuple<std::optional<std::array<Marker3d, 6>>, std::optional<Eigen::Matrix3d>, std::optional<Eigen::Vector<double, 5>>>
readArucoCubeJson(const std::string& jsonFilePath);
}

