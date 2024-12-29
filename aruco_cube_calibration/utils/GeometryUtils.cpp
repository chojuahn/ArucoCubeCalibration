#include "GeometryUtils.hpp"

#include <execution>

#include <Eigen/Dense>
#include <opencv2/imgproc.hpp>   // for cv::kmeans in some OpenCV versions
#include <opencv2/calib3d.hpp>


namespace Util
{
template <typename SizeType>
void forEachIdx(const SizeType loop_size, const std::function<void(const SizeType)>& func) {
#ifdef __APPLE__
    tbb::parallel_for(SizeType(0), loop_size, func);
#else
    std::vector<SizeType> indices(loop_size);
    std::iota(indices.begin(), indices.end(), 0);
    auto execution_mode = std::execution::par;
    std::for_each(execution_mode, indices.begin(), indices.end(), func);
    //std::for_each(indices.begin(), indices.end(), func);
#endif
}
template void forEachIdx<size_t>(const size_t loop_size, const std::function<void(const size_t)>& func);
template void forEachIdx<Eigen::Index>(const Eigen::Index loop_size, const std::function<void(const Eigen::Index)>& func);

Marker3d rotateAboutX(const Marker3d& marker3d, const double angle) {
    Eigen::AngleAxisd rotation(angle, Eigen::Vector3d::UnitX());
    return (rotation.matrix() * marker3d.transpose()).transpose();
}
Marker3d rotateAboutY(const Marker3d& marker3d, double angle) {
    Eigen::AngleAxisd rotation(angle, Eigen::Vector3d::UnitY());
    return (rotation.matrix() * marker3d.transpose()).transpose();
}
Marker3d rotateAboutZ(const Marker3d& marker3d, double angle) {
    Eigen::AngleAxisd rotation(angle, Eigen::Vector3d::UnitZ());
    return (rotation.matrix() * marker3d.transpose()).transpose();
}

std::tuple<cv::Mat, cv::Mat> toCvCameraIntrinsics(const Eigen::Matrix3d& camera_matrix, const Eigen::Vector<double, 5>& distortion_coefs)
{
    cv::Mat camera_matrix_cv(3, 3, CV_64FC1);
    std::copy(camera_matrix.data(), camera_matrix.data() + camera_matrix.size(), reinterpret_cast<double*>(camera_matrix_cv.data));
    cv::transpose(camera_matrix_cv, camera_matrix_cv);

    cv::Mat distortion_coefs_cv(1, 5, CV_64FC1);
    std::copy(distortion_coefs.data(), distortion_coefs.data() + distortion_coefs.size(), reinterpret_cast<double*>(distortion_coefs_cv.data));
    return std::make_tuple(camera_matrix_cv, distortion_coefs_cv);
}
std::tuple<Eigen::Matrix3d, Eigen::Vector<double, 5>> toEigenCameraIntrinsics(const cv::Mat& camera_matrix_cv, const cv::Mat& distortion_coefs_cv)
{
    Eigen::Matrix3d camera_matrix_t;
    std::copy(reinterpret_cast<const double*>(camera_matrix_cv.datastart), reinterpret_cast<const double*>(camera_matrix_cv.dataend), camera_matrix_t.data());

    Eigen::RowVector<double, 5> distortion_coefs_t;
    std::copy(reinterpret_cast<const double*>(distortion_coefs_cv.datastart), reinterpret_cast<const double*>(distortion_coefs_cv.dataend), distortion_coefs_t.data());
    return std::make_tuple(camera_matrix_t.transpose(), distortion_coefs_t.transpose());
}

Eigen::Matrix4d rtToMat4x4(const Eigen::RowVector<double, 6>& rvec)
{
    Eigen::Matrix4d ret = Eigen::Matrix4d::Identity();
    ret.topLeftCorner(3, 3) = rodrigues(rvec.leftCols(3));
    ret.topRightCorner(3, 1) = rvec.rightCols(3).transpose();
    return ret;
}

Eigen::MatrixX2d transform3x3(const Eigen::Matrix3d& mat, const Eigen::MatrixXd& pts)
{
    const auto num_pts = pts.rows();
    Eigen::MatrixXd pts_homogeneous(3, num_pts);
    pts_homogeneous.topRows(2) = pts.transpose();
    pts_homogeneous.row(2) = Eigen::RowVectorXd::Ones(num_pts);

    const Eigen::MatrixX2d ret = (mat * pts_homogeneous).transpose().leftCols(2);
    return ret;
}

Eigen::MatrixX3d transform4x4(const Eigen::Matrix4d& mat, const Eigen::MatrixXd& pts)
{
    const auto num_pts = pts.rows();
    Eigen::MatrixXd pts_homogeneous(4, num_pts);
    pts_homogeneous.topRows(3) = pts.transpose();
    pts_homogeneous.row(3) = Eigen::RowVectorXd::Ones(num_pts);

    const Eigen::MatrixX3d ret = (mat * pts_homogeneous).transpose().leftCols(3);
    return ret;
}

Eigen::Matrix3d rodrigues(const Eigen::Vector3d& rvec) {
    // Calculate the angle (magnitude of the vector)
    const double theta = rvec.norm();
    // If the angle is small, use an approximation for stability
    if (theta < 1e-6) {
        return Eigen::Matrix3d::Identity();
    }
    // Compute the unit vector (axis of rotation)
    const Eigen::Vector3d axis = rvec / theta;
    // Compute the skew-symmetric matrix of the axis
    Eigen::Matrix3d K;
    K << 0, -axis(2), axis(1),
        axis(2), 0, -axis(0),
        -axis(1), axis(0), 0;

    // Use the Rodrigues formula to compute the rotation matrix
    const Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + std::sin(theta) * K + (1 - std::cos(theta)) * K * K;
    return R;
}

std::vector<size_t> getSpatiallyDistributedFrameIndices(const std::vector<Eigen::Vector2d>& frame_markers, size_t n_clusters)
{
    // Limit n_clusters to the number of available points
    n_clusters = std::min(n_clusters, frame_markers.size());

    // If no points or n_clusters = 0, return empty
    if (frame_markers.empty() || n_clusters == 0)
        return {};
    // Special trivial case, return empty
    if (n_clusters == 1 && frame_markers.size() == 1)
        return {};

    // Convert Eigen points to an cv::Mat of shape [N x 2], type float
    const int N = static_cast<int>(frame_markers.size());
    cv::Mat data(N, 2, CV_32F);
    for (int i = 0; i < N; ++i) {
        data.at<float>(i, 0) = static_cast<float>(frame_markers[i].x());
        data.at<float>(i, 1) = static_cast<float>(frame_markers[i].y());
    }

    // Run OpenCV kmeans
    cv::Mat labels;   // will be Nx1, each element in [0..n_clusters-1]
    cv::Mat centers;  // will be n_clusters x 2, each row = cluster centroid

    // K-means criteria: up to 100 iterations or until epsilon < 1.0
    const int attempts = 1;                  // how many times to repeat kmeans with different initial seeds
    const int flags = cv::KMEANS_PP_CENTERS; // initialization method
    cv::TermCriteria criteria(
        cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
        100,    // max iterations
        1.0     // epsilon
    );
    cv::kmeans(data,                    // input data
        static_cast<int>(n_clusters),   // K
        labels,                         // output "best labels" for each sample
        criteria, attempts, flags, centers);

    // For each cluster, find the single point that is closest to the cluster center
    std::vector<size_t> selected_indices;
    selected_indices.reserve(n_clusters);

    // Convert centers to float arrays for distance calculation
    // centers is n_clusters x 2, type CV_32F
    for (int c = 0; c < centers.rows; ++c)
    {
        // Extract the cluster centroid
        const float cx = centers.at<float>(c, 0);
        const float cy = centers.at<float>(c, 1);

        double min_dist = std::numeric_limits<double>::max();
        size_t best_index = 0;
        for (int i = 0; i < N; ++i)
        {
            // Only consider points belonging to cluster 'c'
            // (OpenCV’s kmeans assigns a label in [0..n_clusters-1] to each point)
            if (labels.at<int>(i, 0) == c)
            {
                const double dx = data.at<float>(i, 0) - cx;
                const double dy = data.at<float>(i, 1) - cy;
                const double dist = dx * dx + dy * dy;
                if (dist < min_dist)
                {
                    min_dist = dist;
                    best_index = static_cast<size_t>(i);
                }
            }
        }
        selected_indices.push_back(best_index);
    }

    return selected_indices;
}

std::tuple<Eigen::Vector3d, Eigen::Vector3d> getPointcloudPose(
    const Eigen::MatrixX3d& pointcloud_local,
    const Eigen::MatrixX2d& pointcloud_image,
    const Eigen::Matrix3d& camera_matrix,
    const Eigen::Vector<double, 5>& distortion_coefs)
{
    // 3D points: Nx1, CV_64FC3
    const int N = static_cast<int>(pointcloud_local.rows());
    cv::Mat object_points(N, 1, CV_64FC3);  // N x 1 with 3 channels
    for (int i = 0; i < N; ++i) {
        object_points.at<cv::Vec3d>(i, 0) = cv::Vec3d(
            pointcloud_local(i, 0),
            pointcloud_local(i, 1),
            pointcloud_local(i, 2)
        );
    }
    // 2D points: Nx1, CV_64FC2
    cv::Mat image_points(N, 1, CV_64FC2);
    for (int i = 0; i < N; ++i) {
        image_points.at<cv::Vec2d>(i, 0) = cv::Vec2d(
            pointcloud_image(i, 0),
            pointcloud_image(i, 1)
        );
    }
    // Camera matrix: 3x3, CV_64F
    cv::Mat camera_mat(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            camera_mat.at<double>(r, c) = camera_matrix(r, c);
        }
    }
    // Distortion coefficients: 1x5, CV_64F
    cv::Mat dist_coeffs(1, 5, CV_64F);
    for (int i = 0; i < 5; ++i) {
        dist_coeffs.at<double>(0, i) = distortion_coefs(i);
    }

    // Solve PnP initial estimate
    cv::Mat rvec, tvec;
    const auto method = (N >= 4 ? cv::SolvePnPMethod::SOLVEPNP_ITERATIVE : cv::SolvePnPMethod::SOLVEPNP_P3P);
    const bool success = cv::solvePnP(
        object_points, image_points, camera_mat, dist_coeffs,
        rvec, tvec, false, method);

    if (!success)
        std::cout << "[getPointcloudPose] solvePnP failed" << std::endl;

    // Refine with solvePnPRefineLM
	cv::solvePnPRefineLM(object_points, image_points, camera_mat, dist_coeffs, rvec, tvec, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 100, 1e-7));

    // Return final rvec, tvec
    const Eigen::Vector3d rvec_eig(rvec.at<double>(0, 0), rvec.at<double>(1, 0), rvec.at<double>(2, 0));
    const Eigen::Vector3d tvec_eig(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0));
    return std::make_pair(rvec_eig, tvec_eig);
}

std::array<Marker3d, 6> alignCornersToOrthogonalFrame(const std::array<Marker3d, 6>& corners_cube_6x4x3)
{
    // Flatten the input into a 24×3 matrix (since we have 6 faces × 4 corners = 24 points).
    Eigen::Matrix<double, 24, 3> all_points;
    for (Eigen::Index i = 0; i < 6; ++i)
        for (Eigen::Index j = 0; j < 4; ++j)
            all_points.row(i * 4 + j) = corners_cube_6x4x3[i].row(j);

    // Compute the centroid (mean of each column).
    const Eigen::RowVector3d centroid = all_points.colwise().mean();

    // Center all points by subtracting the centroid.
    Eigen::Matrix<double, 24, 3> centered_points;
    for (Eigen::Index i = 0; i < 24; ++i)
        centered_points.row(i) = all_points.row(i) - centroid;

    // The "front" corners are the first 4 rows in the flattened array,
    // which corresponds to corners_cube_6x4x3[0] in the original array.
    const Marker3d front_corners = centered_points.topRows(4);

    // Compute the front normal (Z-direction):
    // front_normal = cross( (C0 - C1), (C2 - C1) ), then normalize.
    const Eigen::RowVector3d v1 = front_corners.row(0) - front_corners.row(1);
    const Eigen::RowVector3d v2 = front_corners.row(2) - front_corners.row(1);
    const Eigen::RowVector3d front_normal = v1.cross(v2).normalized();

    // Compute front_x as (C0 - C1) normalized.
    const Eigen::RowVector3d front_x = v1.normalized();

    // Compute front_y = cross(front_normal, front_x), then normalize.
    const Eigen::RowVector3d front_y = front_normal.cross(front_x).normalized();

    // Build the rotation matrix R from these three orthogonal vectors:
    // columns = [front_x, front_y, front_normal].
    Eigen::Matrix3d R;
    R.col(0) = front_x.transpose();
    R.col(1) = front_y.transpose();
    R.col(2) = front_normal.transpose();

    // Rotate all centered points.
    Eigen::Matrix<double, 24, 3> aligned_points = centered_points * R;

    // Reshape the 24×3 back into the std::array<Eigen::Matrix<double, 4, 3>, 6>.
    std::array<Marker3d, 6> aligned_corners;
    for (Eigen::Index i = 0; i < 6; ++i)
        for (Eigen::Index j = 0; j < 4; ++j)
            aligned_corners[i].row(j) = aligned_points.row(i * 4 + j);

    return aligned_corners;
}
}
