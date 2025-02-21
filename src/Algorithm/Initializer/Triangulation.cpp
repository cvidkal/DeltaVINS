#include "Algorithm/Initializer/Triangulation.h"

#include "utils/utils.h"

namespace DeltaVins {

/**
 * @brief Triangulate a point in world coordinate system
 * @param ray_in_c: ray in current frame
 * @param Rwc: rotation matrix from world to current frame
 * @param Pc_in_w: position of current frame in world coordinate system
 * @param Pw_triangulated: position of triangulated point in world coordinate
 * system
 * @return true if triangulation is successful
 * @note X : Point to be triangulated in anchor frame
 * @note ray_i : i index ray transformed to anchor frame
 * @note Pc_i_in_anchor : position of i index frame in anchor frame
 * @note Main equation: ray_i x (X - Pc_i_in_anchor) = 0
 * @cite OpenVINS https://github.com/rpng/open_vins
 */

double max_condition_number = 1e5;

bool TriangulationXYZ(const std::vector<Eigen::Vector3d>& ray_in_c,
                      const std::vector<Eigen::Matrix3d>& Rwc,
                      const std::vector<Eigen::Vector3d>& Pc_in_w,
                      Eigen::Vector3d& Pw_triangulated) {
    Eigen::Matrix3d A;
    Eigen::Vector3d b;

    Eigen::Matrix3d Rwc_anchor = Rwc[0].cast<double>();
    Eigen::Vector3d Pc_anchor = Pc_in_w[0].cast<double>();

    A.setZero();
    b.setZero();

    for (size_t i = 0; i < ray_in_c.size(); i++) {
        // Compute Relative Transformations between anchor and current frame
        Eigen::Matrix3d RA_c = Rwc_anchor.transpose() * Rwc[i].cast<double>();
        Eigen::Vector3d Pc_in_A =
            Rwc_anchor.transpose() * (Pc_in_w[i].cast<double>() - Pc_anchor);

        Eigen::Vector3d ray_in_A = RA_c * ray_in_c[i].cast<double>();

        Eigen::Matrix3d cross_ray_A = crossMat(ray_in_A);

        Eigen::Matrix3d corss_ray_A_2 = cross_ray_A.transpose() * cross_ray_A;

        A.noalias() += corss_ray_A_2;
        b.noalias() += corss_ray_A_2 * Pc_in_A;
    }

    Eigen::Vector3d Pw_triangulated_d = A.ldlt().solve(b);
    // Check condition number of A
    double cond_num =
        A.ldlt().vectorD().maxCoeff() / A.ldlt().vectorD().minCoeff();
    if (cond_num > max_condition_number ||
        std::isnan(Pw_triangulated_d.sum())) {
        return false;
    }

    Pw_triangulated = Rwc_anchor * Pw_triangulated_d + Pc_anchor;

    return true;
}

/**
 * @brief Triangulate a point in world coordinate system
 * @param ray_in_c: ray in current frame
 * @param Rwc: rotation matrix from world to current frame
 * @param Pc_in_w: position of current frame in world coordinate system
 * @param depth: depth of triangulated point
 * @return true if triangulation is successful
 * @note d : depth of triangulated point of anchor ray
 * @note ray_i : i index ray transformed to anchor frame
 * @note Pc_i_in_anchor : position of i index frame in anchor frame
 * @note ray_anchor : anchor ray
 * @note Main equation: ray_i x (d * ray_anchor - Pc_anchor) = 0
 */

bool TriangulationAnchorDepth(const std::vector<Eigen::Vector3d>& ray_in_c,
                              const std::vector<Eigen::Matrix3d>& Rwc,
                              const std::vector<Eigen::Vector3d>& Pc_in_w,
                              Eigen::Vector3d& Pw_triangulated) {
    double A = 0;
    double b = 0;

    Eigen::Matrix3d Rwc_anchor = Rwc[0];
    Eigen::Vector3d Pc_anchor = Pc_in_w[0];
    Eigen::Vector3d ray_anchor = ray_in_c[0];

    for (size_t i = 0; i < ray_in_c.size(); i++) {
        Eigen::Matrix3d RA_c = Rwc_anchor.transpose() * Rwc[i];
        Eigen::Vector3d Pc_in_A =
            Rwc_anchor.transpose() * (Pc_in_w[i] - Pc_anchor);

        Eigen::Vector3d ray_in_A = RA_c * ray_in_c[i];

        Eigen::Matrix3d cross_ray_A = crossMat(ray_in_A);

        Eigen::Vector3d cross_ray_A_ray_anchor = cross_ray_A * ray_anchor;

        A += cross_ray_A_ray_anchor.transpose() * cross_ray_A_ray_anchor;
        b += cross_ray_A_ray_anchor.transpose() * cross_ray_A * Pc_in_A;
    }

    double depth = b / A;

    if (std::isnan(depth)) {
        return false;
    }

    Pw_triangulated = Rwc_anchor * (ray_anchor * depth) + Pc_anchor;

    return true;
}

}  // namespace DeltaVins
