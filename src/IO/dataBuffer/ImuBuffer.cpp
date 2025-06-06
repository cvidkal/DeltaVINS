/**
 * This file is part of Delta_VIO.
 *
 * Delta_VIO is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Delta_VIO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Delta_VIO. If not, see <http://www.gnu.org/licenses/>.
 */
#include "IO/dataBuffer/imuBuffer.h"

#include <sophus/se3.hpp>

#include "Algorithm/IMU/ImuPreintergration.h"
#include "precompile.h"
#include "utils/SensorConfig.h"
#include "utils/utils.h"

namespace DeltaVins {
ImuBuffer::ImuBuffer() : CircularBuffer<ImuData, 10>() {
    gyro_bias_.setZero();
    acc_bias_.setZero();

    IMUParams imuParams = SensorConfig::Instance().GetIMUParams(0);
    static const float gyro_noise = imuParams.gyro_noise;
    static const float acc_noise = imuParams.acc_noise;
    static const int imu_fps = imuParams.fps;
    static const float gyro_noise2 = gyro_noise * (gyro_noise * imu_fps);
    static const float acc_noise2 = acc_noise * (acc_noise * imu_fps);

    noise_cov_.setIdentity(6, 6);
    noise_cov_.topLeftCorner(3, 3) *= gyro_noise2;
    noise_cov_.bottomRightCorner(3, 3) *= acc_noise2;

    gravity_.setZero();
}

void ImuBuffer::UpdateBias(const Vector3f& dBg, const Vector3f& dBa) {
    gyro_bias_ += dBg;
    acc_bias_ += dBa;
}

void ImuBuffer::SetBias(const Vector3f& bg, const Vector3f& ba) {
    gyro_bias_ = bg;
    acc_bias_ = ba;
}

void ImuBuffer::SetZeroBias() {
    gyro_bias_.setZero();
    acc_bias_.setZero();
}

void ImuBuffer::GetBias(Vector3f& bg, Vector3f& ba) const {
    bg = gyro_bias_;
    ba = acc_bias_;
}

Vector3f ImuBuffer::GetGravity() {
    std::lock_guard<std::mutex> lck(gravity_mutex_);
    return gravity_;
}

bool ImuBuffer::GetDataByBinarySearch(ImuData& imuData) const {
    int index = binarySearch<long long>(imuData.timestamp, Left);
    if (index < 0) {
        LOGE("t:%lld,imu0:%lld,imu1:%lld\n", (long long)imuData.timestamp,
             (long long)buf_[getDeltaIndex(tail_, 3)].timestamp,
             (long long)buf_[getDeltaIndex(head_, -1)].timestamp);

        throw std::runtime_error("No Imu data found,Please check timestamp1");
    }

    auto& left = buf_[index];
    auto& right = buf_[getDeltaIndex(index, 1)];

    // linear interpolation
    float k = float(imuData.timestamp - left.timestamp) /
              float(right.timestamp - left.timestamp);

    imuData.gyro = linearInterpolate(left.gyro, right.gyro, k);
    imuData.acc = linearInterpolate(left.acc, right.acc, k);

    return index >= 0;
}

inline Matrix3f vector2Jac(const Vector3f& x) {
    return Matrix3f::Identity() - 0.5f * crossMat(x);
}

/*----------------------------------------------------------------------------
 * IMU Preintegration on Manifold for Efficient Visual-Inertial
 * Maximum-a-Posteriori Estimation"
 * http://www.roboticsproceedings.org/rss11/p06.pdf
 */
bool ImuBuffer::ImuPreIntegration(ImuPreintergration& ImuTerm) const {
    if (ImuTerm.t0 >= ImuTerm.t1) {
        LOGW("t0:%lld t1:%lld", (long long)ImuTerm.t0, (long long)ImuTerm.t1);
        throw std::runtime_error("t0>t1");
    }
    int Index0 = binarySearch<long long>(ImuTerm.t0, Left);
    int Index1 = binarySearch<long long>(ImuTerm.t1, Left);

    int try_times = 0;
    while (Index1 < 0) {
        LOGI("t1:%lld,imu1:%lld", (long long)ImuTerm.t1,
             (long long)buf_[getDeltaIndex(head_, -1)].timestamp);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        Index1 = binarySearch<long long>(ImuTerm.t1, Left);
        try_times++;
        if (try_times > 20 || Config::SerialRun) {
            throw std::runtime_error(
                "IMU is slower than Image, waiting for IMU data...");
        }
    }
    // if(Index1 < 0){
    //     Index1 = getDeltaIndex(head_, -1);
    // }

    if (Index0 < 0 || Index1 < 0) {
        LOGW("dt0:%lld dt1:%lld,dT:%lld",
             (long long)ImuTerm.t0 - buf_[Index0].timestamp,
             (long long)ImuTerm.t1 - buf_[Index1].timestamp,
             (long long)ImuTerm.t1 - ImuTerm.t0);
        if (Index0 < 0) {
            LOGW("Error Code:%d", Index0);
            LOGE("t0:%lld,imu0:%lld\n", (long long)ImuTerm.t0,
                 (long long)buf_[getDeltaIndex(tail_, 3)].timestamp);
        }
        if (Index1 < 0) {
            LOGE("t1:%lld,imu1:%lld\n", (long long)ImuTerm.t1,
                 (long long)buf_[getDeltaIndex(head_, -1)].timestamp);
        }
        throw std::runtime_error("No Imu data found.Please check timestamp2");
    }
    ImuTerm.reset();

    Matrix3f& dRdg = ImuTerm.dRdg;
    Matrix3f& dPda = ImuTerm.dPda;
    Matrix3f& dPdg = ImuTerm.dPdg;
    Matrix3f& dVda = ImuTerm.dVda;
    Matrix3f& dVdg = ImuTerm.dVdg;

    Matrix3f& dR = ImuTerm.dR;
    Vector3f& dV = ImuTerm.dV;
    Vector3f& dP = ImuTerm.dP;
    Matrix3f dR0 = dR;
    Vector3f dV0 = dV;

    Matrix9f& cov = ImuTerm.Cov;

    MatrixXf A = MatrixXf::Identity(9, 9);
    MatrixXf B = MatrixXf::Zero(9, 6);

    int indexEnd = getDeltaIndex(Index1, 1);

    int index = Index0;

    float dt;

    Vector3f gyro, acc;

    while (index != indexEnd) {
        auto& imuData = buf_[index];
        int nextIndex = getDeltaIndex(index, 1);
        auto& nextImuData = buf_[nextIndex];
        ImuTerm.sensor_id = imuData.sensor_id;

        if (index == Index0) {
            dt = nextImuData.timestamp - ImuTerm.t0;
            float a = nextImuData.timestamp - ImuTerm.t0;
            float b = nextImuData.timestamp - imuData.timestamp;
            float k = a / b;
            gyro = linearInterpolate(imuData.gyro, nextImuData.gyro, 1 - k);
            acc = linearInterpolate(imuData.acc, nextImuData.acc, 1 - k);
        } else if (index == Index1) {
            dt = ImuTerm.t1 - imuData.timestamp;
            float a = nextImuData.timestamp - ImuTerm.t1;
            float b = nextImuData.timestamp - imuData.timestamp;
            float k = a / b;
            gyro = linearInterpolate(imuData.gyro, nextImuData.gyro, k);
            acc = linearInterpolate(imuData.acc, nextImuData.acc, k);
        } else {
            dt = nextImuData.timestamp - imuData.timestamp;
            gyro = (imuData.gyro + nextImuData.gyro) * 0.5f;
            acc = (imuData.acc + nextImuData.acc) * 0.5f;
        }
        dt *= 1e-9;

        gyro -= gyro_bias_;
        acc -= acc_bias_;

        Vector3f ddV0 = acc * dt;
        Vector3f ddR0 = gyro * dt;

        Matrix3f ddR = Sophus::SO3Group<float>::exp(ddR0).matrix();

        // update covariance iteratively
        A.topLeftCorner(3, 3) = ddR.transpose();
        A.block<3, 3>(3, 0) = -dR0 * crossMat(ddV0);
        A.bottomLeftCorner(3, 3) = 0.5 * dt * A.block<3, 3>(3, 0);
        A.block<3, 3>(6, 3) = Matrix3f::Identity() * dt;

        B.topLeftCorner(3, 3) = vector2Jac(ddR0) * dt;
        B.block<3, 3>(3, 3) = dR0 * dt;
        B.block<3, 3>(6, 3) = dR0 * (0.5 * dt * dt);

        cov = A * cov * A.transpose() + B * noise_cov_ * B.transpose();

        // update Jacobian Matrix iteratively
        dRdg -= ddR.transpose() * vector2Jac(ddR0) * dt;
        dVda -= dR0 * dt;
        dVdg -= dR0 * crossMat(ddV0) * dRdg;
        dPda -= 1.5f * dR0 * dt * dt;
        dPdg -= 1.5f * dR0 * crossMat(ddV0) * dRdg * dt;

        // update delta states iteratively
        Vector3f ddV = dR0 * acc * dt;
        dP += dV0 * dt + 0.5f * ddV * dt;
        dR0 = dR = dR0 * ddR;
        dV0 = dV = dV0 + ddV;

        index = nextIndex;
    }

    // add time
    ImuTerm.dT += ImuTerm.t1 - ImuTerm.t0;

    static const int64_t max_dt =
        1e9 / SensorConfig::Instance().GetCameraParams(0).fps * 1.5;
    if (ImuTerm.dT > max_dt) {
        LOGW("Detected a Frame Drop, dT:%lld max_dt:%lld",
             (long long)ImuTerm.dT, (long long)max_dt);
    }

    return true;
}

void ImuBuffer::OnImuReceived(const ImuData& imuData) {
    buf_[head_] = imuData;

    // do low pass filter to get gravity
    static Eigen::Vector3f gravity = imuData.acc;
    gravity = 0.95f * gravity + 0.05f * imuData.acc;
    {
        std::lock_guard<std::mutex> lck(gravity_mutex_);
        gravity_ = gravity;
    }

    PushIndex();
}

Vector3f ImuBuffer::GetGravity(long long timestamp) {
    return gravity_;
    BufferIndex index0 = binarySearch<long long>(timestamp, Left);
    if (index0 < 0) {
        std::lock_guard<std::mutex> lck(gravity_mutex_);
        return gravity_;
    }
    int nSize = index0 > tail_ ? index0 - tail_ : index0 + _END - tail_;
    BufferIndex index_start =
        getDeltaIndex(index0, nSize >= 21 ? -20 : -nSize + 1);
    BufferIndex index_end = index0;
    BufferIndex i = index_start;
    Vector3f gravity = buf_[i].acc;
    while (i != index_end) {
        gravity = 0.95f * gravity + 0.05f * buf_[i].acc;
        i = getDeltaIndex(i, 1);
    }
    return gravity;
}

void ImuBuffer::UpdateBiasByStatic(long long timestamp) {
    BufferIndex index = binarySearch(timestamp, Left);
    index = getDeltaIndex(head_, -1);
    int nSize = index > tail_ ? index - tail_ : index + _END - tail_;
    BufferIndex index_start =
        getDeltaIndex(index, nSize > 100 ? -100 : -nSize + 1);
    Vector3f sum_gyro(0, 0, 0);
    for (int i = 0; i < nSize; i++) {
        sum_gyro += buf_[getDeltaIndex(index_start, i)].gyro;
    }
    Vector3f mean_gyro = sum_gyro / nSize;
    SetBias(mean_gyro, Vector3f(0, 0, 0));
    LOGI("Update bias by static, mean_gyro:%f %f %f", mean_gyro.x(),
         mean_gyro.y(), mean_gyro.z());
}

bool ImuBuffer::DetectStatic(long long timestamp) const {
    BufferIndex index1 = binarySearch(timestamp, Left);
    int nSize = index1 > tail_ ? index1 - tail_ : index1 + _END - tail_;
    if (nSize < 100) return false;

    nSize = nSize < 200 ? nSize : 200;

    BufferIndex index0 = getDeltaIndex(index1, -nSize);
    Vector3f sum_acc(0, 0, 0);
    Vector3f sum_gyro(0, 0, 0);
    for (int i = 0; i < nSize; ++i) {
        auto& imu_data = buf_[getDeltaIndex(index0, i)];
        sum_acc += imu_data.acc;
        sum_gyro += imu_data.gyro;
    }
    Vector3f mean_acc = sum_acc / nSize;
    Vector3f mean_gyro = sum_gyro / nSize;

    float g_div = 0;
    float a_div = 0;

    for (int i = 0; i < nSize; ++i) {
        auto& imu_data = buf_[getDeltaIndex(index0, i)];
        a_div += (imu_data.acc - mean_acc).norm();
        g_div += (imu_data.gyro - mean_gyro).norm();
    }
    a_div /= nSize;
    g_div /= nSize;

    float g_div_thresh = 0.04;
    float a_div_thresh = 0.5;

    if (g_div < g_div_thresh && a_div < a_div_thresh) return true;
    return false;
}

}  // namespace DeltaVins
