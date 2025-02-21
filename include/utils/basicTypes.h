//
// Created by chenguojun on 2020/6/10.
//

#pragma once
#include <Eigen/Eigen>

namespace DeltaVins {

constexpr auto EigenMajorType = Eigen::ColMajor;

template <typename T>
using Vector2 = Eigen::Matrix<T, 2, 1, EigenMajorType>;

using Vector2f = Vector2<float>;

template <typename T>
using Vector3 = Eigen::Matrix<T, 3, 1, EigenMajorType>;
using Vector3f = Vector3<float>;

template <typename T>
using Matrix2 = Eigen::Matrix<T, 2, 2, EigenMajorType>;
using Matrix2f = Matrix2<float>;

template <typename T>
using Matrix23 = Eigen::Matrix<T, 2, 3, EigenMajorType>;
using Matrix23f = Matrix23<float>;

template <typename T>
using Matrix3 = Eigen::Matrix<T, 3, 3, EigenMajorType>;
using Matrix3f = Matrix3<float>;

template <typename T>
using Matrix9 = Eigen::Matrix<T, 9, 9, EigenMajorType>;
using Matrix9f = Matrix9<float>;

template <typename T>
using Quaternion = Eigen::Quaternion<T>;
using Quaternionf = Quaternion<float>;

using Vector2d = Vector2<double>;

using Vector3d = Vector3<double>;

using Matrix2d = Matrix2<double>;

using Matrix23d = Matrix23<double>;

using Matrix3d = Matrix3<double>;

using MatrixXd = Eigen::Matrix<double, -1, -1, EigenMajorType>;
using VectorXd = Eigen::Matrix<double, -1, 1, EigenMajorType>;

using Quaterniond = Quaternion<double>;

using MatrixXf = Eigen::Matrix<float, -1, -1, EigenMajorType>;
using VectorXf = Eigen::Matrix<float, -1, 1, EigenMajorType>;
}  // namespace DeltaVins