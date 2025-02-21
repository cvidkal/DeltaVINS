#pragma once
#include "dataStructure/vioStructures.h"

namespace DeltaVins {

bool TriangulationXYZ(const std::vector<Eigen::Vector3d>& ray_in_c,
                      const std::vector<Eigen::Matrix3d>& Rwc,
                      const std::vector<Eigen::Vector3d>& Pc_in_w,
                      Eigen::Vector3d& Pw_triangulated);
bool TriangulationAnchorDepth(const std::vector<Eigen::Vector3d>& ray_in_c,
                              const std::vector<Eigen::Matrix3d>& Rwc,
                              const std::vector<Eigen::Vector3d>& Pc_in_w,
                              Eigen::Vector3d& Pw_triangulated);

};  // namespace DeltaVins