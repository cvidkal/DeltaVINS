#include "precompile.h"
#include <Algorithm/Initializer/Triangulation.h>
#include <gtest/gtest.h>

// generate Triangulation unit test

TEST(Triangulation, TriangulationXYZ_DataVerification) {
    // generate Pw
    Eigen::Vector3f Pw(10.0, 10.0, 30.0);

    // generate Rwc with small rotation
    std::vector<Eigen::Matrix3f> Rwc;
    Rwc.push_back(Eigen::Matrix3f::Identity());
    Rwc.push_back((Eigen::AngleAxisf(0.1, Eigen::Vector3f::UnitX()) *
                   Eigen::AngleAxisf(0.1, Eigen::Vector3f::UnitY()) *
                   Eigen::AngleAxisf(0.1, Eigen::Vector3f::UnitZ()))
                      .matrix());
    Rwc.push_back((Eigen::AngleAxisf(0.2, Eigen::Vector3f::UnitX()) *
                   Eigen::AngleAxisf(0.2, Eigen::Vector3f::UnitY()) *
                   Eigen::AngleAxisf(0.2, Eigen::Vector3f::UnitZ()))
                      .matrix());
    Rwc.push_back((Eigen::AngleAxisf(0.3, Eigen::Vector3f::UnitX()) *
                   Eigen::AngleAxisf(0.3, Eigen::Vector3f::UnitY()) *
                   Eigen::AngleAxisf(0.3, Eigen::Vector3f::UnitZ()))
                      .matrix());

    // generate Pc_in_w
    std::vector<Eigen::Vector3f> Pc_in_w;
    Pc_in_w.push_back(Eigen::Vector3f(1.0, 2.0, 3.0));
    Pc_in_w.push_back(Eigen::Vector3f(1.0, 3.0, 3.0));
    Pc_in_w.push_back(Eigen::Vector3f(2.0, 2.0, 3.0));
    Pc_in_w.push_back(Eigen::Vector3f(5.0, 2.0, 3.0));

    // generate ray_in_c
    std::vector<Eigen::Vector3f> ray_in_c;
    for (int i = 0; i < 4; i++) {
        Eigen::Vector3f ray = Rwc[i].transpose() * (Pw - Pc_in_w[i]);
        ray *= 1.0 / ray.z();
        ray_in_c.push_back(ray);
    }

    // Eigen::Vector3f ray_anchor = ray_in_c[0];
    Eigen::Vector3f Pw_anchor = Rwc[0].transpose() * (Pw - Pc_in_w[0]);
    // Call TriangulationXYZ
    for (int i = 0; i < 4; i++) {
        Eigen::Matrix3f RA_c = Rwc[0].transpose() * Rwc[i];
        Eigen::Vector3f Pc_in_A =
            Rwc[0].transpose() * (Pc_in_w[i] - Pc_in_w[0]);
        Eigen::Vector3f ray = Pw_anchor - Pc_in_A;
        Eigen::Vector3f ray_in_A = RA_c * ray_in_c[i];
        ray *= 1.0 / ray.z();

        Eigen::Matrix3f cross_ray_A = DeltaVins::crossMat(ray);
        Eigen::Vector3f result = cross_ray_A * ray_in_A;
        EXPECT_NEAR(result.norm(), 0.0, 1e-6);
    }
}

TEST(Triangulation, TriangulationXYZ) {
    // generate Pw
    Eigen::Vector3d Pw(10.0, 10.0, 30.0);

    // generate Rwc with small rotation
    std::vector<Eigen::Matrix3d> Rwc;
    Rwc.push_back(Eigen::Matrix3d::Identity());
    Rwc.push_back((Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitX()) *
                   Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitY()) *
                   Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitZ()))
                      .matrix());
    Rwc.push_back((Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitX()) *
                   Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY()) *
                   Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitZ()))
                      .matrix());
    Rwc.push_back((Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitX()) *
                   Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY()) *
                   Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ()))
                      .matrix());

    // generate Pc_in_w
    std::vector<Eigen::Vector3d> Pc_in_w;
    Pc_in_w.push_back(Eigen::Vector3d(1.0, 2.0, 3.0));
    Pc_in_w.push_back(Eigen::Vector3d(1.0, 3.0, 3.0));
    Pc_in_w.push_back(Eigen::Vector3d(2.0, 2.0, 3.0));
    Pc_in_w.push_back(Eigen::Vector3d(5.0, 2.0, 3.0));

    // generate ray_in_c
    std::vector<Eigen::Vector3d> ray_in_c;
    for (int i = 0; i < 4; i++) {
        Eigen::Vector3d ray = Rwc[i].transpose() * (Pw - Pc_in_w[i]);
        ray *= 1.0 / ray.z();
        ray_in_c.push_back(ray);
    }

    // Call TriangulationXYZ
    Eigen::Vector3d Pw_triangulated;
    bool success =
        DeltaVins::TriangulationXYZ(ray_in_c, Rwc, Pc_in_w, Pw_triangulated);
    EXPECT_TRUE(success);
    EXPECT_NEAR(Pw_triangulated.x(), Pw.x(), 1e-6);
    EXPECT_NEAR(Pw_triangulated.y(), Pw.y(), 1e-6);
    EXPECT_NEAR(Pw_triangulated.z(), Pw.z(), 1e-6);
}

TEST(Triangulation, TriangulationAnchorDepth) {
    // generate Pw
    Eigen::Vector3d Pw(10.0, 10.0, 30.0);

    // generate Rwc with small rotation
    std::vector<Eigen::Matrix3d> Rwc;
    Rwc.push_back(Eigen::Matrix3d::Identity());
    Rwc.push_back((Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitX()) *
                   Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitY()) *
                   Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitZ()))
                      .matrix());
    Rwc.push_back((Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitX()) *
                   Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY()) *
                   Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitZ()))
                      .matrix());
    Rwc.push_back((Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitX()) *
                   Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY()) *
                   Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ()))
                      .matrix());

    // generate Pc_in_w
    std::vector<Eigen::Vector3d> Pc_in_w;
    Pc_in_w.push_back(Eigen::Vector3d(1.0, 2.0, 3.0));
    Pc_in_w.push_back(Eigen::Vector3d(1.0, 3.0, 3.0));
    Pc_in_w.push_back(Eigen::Vector3d(2.0, 2.0, 3.0));
    Pc_in_w.push_back(Eigen::Vector3d(5.0, 2.0, 3.0));

    // generate ray_in_c
    std::vector<Eigen::Vector3d> ray_in_c;
    for (int i = 0; i < 4; i++) {
        Eigen::Vector3d ray = Rwc[i].transpose() * (Pw - Pc_in_w[i]);
        ray *= 1.0 / ray.z();
        ray_in_c.push_back(ray);
    }

    // Call TriangulationAnchorDepth
    Eigen::Vector3d Pw_triangulated;
    bool success = DeltaVins::TriangulationAnchorDepth(ray_in_c, Rwc, Pc_in_w,
                                                       Pw_triangulated);
    EXPECT_TRUE(success);
    EXPECT_NEAR(Pw_triangulated.x(), Pw.x(), 1e-6);
    EXPECT_NEAR(Pw_triangulated.y(), Pw.y(), 1e-6);
    EXPECT_NEAR(Pw_triangulated.z(), Pw.z(), 1e-6);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}