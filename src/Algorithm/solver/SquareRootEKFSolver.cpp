#include "Algorithm/solver/SquareRootEKFSolver.h"

#include <sophus/so3.hpp>

#include "Algorithm/IMU/ImuPreintergration.h"
#include "Algorithm/vision/camModel/camModel.h"
#include "IO/dataBuffer/imuBuffer.h"
#include "precompile.h"
#include "utils/SensorConfig.h"
#include "utils/TickTock.h"
#include "utils/constantDefine.h"
#include "utils/utils.h"

namespace DeltaVins {
SquareRootEKFSolver::SquareRootEKFSolver() {}

void SquareRootEKFSolver::Init(CamState* state, Vector3f* vel, bool* static_) {
    VectorXf p(NEW_STATE_DIM);

    switch (Config::DataSourceType) {
        case DataSrcEuroc:
            p << 1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2, 1e-3, 1e-3, 1e-3, 1e-4,
                1e-4, 1e-4, 1e-2, 1e-2, 1e-2;
            break;
        default:
            p << 1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2, 1e-3, 1e-3, 1e-3, 1e-4,
                1e-4, 1e-4, 1e-2, 1e-2, 1e-2;
            break;
    }

    CURRENT_DIM = NEW_STATE_DIM;

    vel_ = vel;
    static_ = static_;
    info_factor_matrix_.topLeftCorner(CURRENT_DIM, CURRENT_DIM) =
        p.cwiseSqrt().cwiseInverse().asDiagonal();

    vel_->setZero();
    AddCamState(state);
    cam_states_.clear();
    cam_states_.push_back(state);
}

void SquareRootEKFSolver::AddCamState(CamState* state) {
    last_state_ = new_state_;
    new_state_ = state;
}
void SquareRootEKFSolver::PropagateStatic(const ImuPreintergration* imu_term) {
    static Vector3f gravity(0, 0, GRAVITY);
    Matrix3f R0 = last_state_->Rwi;

    float dt = imu_term->dT * 1e-9;

    // propagate states
    new_state_->Pwi = last_state_->Pwi + (R0 * imu_term->dP + *vel_ * dt +
                                          gravity * (0.5f * dt * dt));
    new_state_->Pw_FEJ = new_state_->Pwi;
    *vel_ += R0 * imu_term->dV + gravity * dt;
    new_state_->Rwi = R0 * imu_term->dR;

    // Make state transition matrix F
    static Eigen::Matrix<float, NEW_STATE_DIM, NEW_STATE_DIM>
        state_transition_matrix;
    state_transition_matrix.setIdentity();
    state_transition_matrix.block<3, 3>(0, 0) = imu_term->dR.transpose();
    state_transition_matrix.block<3, 3>(0, 6) = imu_term->dRdg;
    state_transition_matrix.block<3, 3>(3, 0) = -R0 * crossMat(imu_term->dP);
    state_transition_matrix.block<3, 3>(3, 6) = R0 * imu_term->dPdg;
    state_transition_matrix.block<3, 3>(3, 9) = Matrix3f::Identity() * dt;
    state_transition_matrix.block<3, 3>(9, 0) = -R0 * crossMat(imu_term->dV);
    state_transition_matrix.block<3, 3>(9, 6) = R0 * imu_term->dVdg;
    state_transition_matrix.block<3, 3>(9, 12) = R0 * imu_term->dVda;
    state_transition_matrix.block<3, 3>(3, 12) = R0 * imu_term->dPda;

    // make noise transition matrix
    static Eigen::Matrix<float, NEW_STATE_DIM, 9> noise_transition_matrix;
    noise_transition_matrix.setZero();
    noise_transition_matrix.block<3, 3>(0, 0).setIdentity();
    noise_transition_matrix.block<3, 3>(3, 6) = R0;
    noise_transition_matrix.block<3, 3>(9, 3) = R0;

    // Make Noise Covariance Matrix Q
    static Eigen::Matrix<float, NEW_STATE_DIM, NEW_STATE_DIM> noise_cov;
    const float gyro_bias_noise =
        SensorConfig::Instance().GetIMUParams(imu_term->sensor_id).gyro_noise;
    const float acc_bias_noise =
        SensorConfig::Instance().GetIMUParams(imu_term->sensor_id).acc_noise;
    const int nImuSample =
        SensorConfig::Instance().GetIMUParams(imu_term->sensor_id).fps;
    const float gyro_bias_noise2 =
        gyro_bias_noise * (gyro_bias_noise * nImuSample);
    const float acc_bias_noise2 =
        acc_bias_noise * (acc_bias_noise * nImuSample);
    noise_cov = noise_transition_matrix * imu_term->Cov *
                noise_transition_matrix.transpose();
    noise_cov.block<3, 3>(6, 6) =
        Matrix3f::Identity() * (gyro_bias_noise2 * dt * dt);
    noise_cov.block<3, 3>(12, 12) =
        Matrix3f::Identity() * (acc_bias_noise2 * dt * dt);

    static Eigen::Matrix<float, NEW_STATE_DIM, NEW_STATE_DIM> NoiseFactor;
    NoiseFactor.setIdentity();
    Eigen::LLT<MatrixXf> chol(noise_cov);
    chol.matrixU().solveInPlace(NoiseFactor);

    // Make information factor matrix
    int OLD_DIM = CURRENT_DIM;
    CURRENT_DIM += NEW_STATE_DIM;

    int IMUIdx = 0;
    info_factor_matrix_.block(0, OLD_DIM, OLD_DIM, NEW_STATE_DIM).setZero();
    info_factor_matrix_.block(OLD_DIM, 0, NEW_STATE_DIM, OLD_DIM).setZero();
    MatrixXf R = NoiseFactor * state_transition_matrix;
    info_factor_matrix_.block<NEW_STATE_DIM, IMU_STATE_DIM>(OLD_DIM, IMUIdx) =
        R.rightCols<IMU_STATE_DIM>();
    info_factor_matrix_.block<NEW_STATE_DIM, CAM_STATE_DIM>(
        OLD_DIM, OLD_DIM - CAM_STATE_DIM) = R.leftCols<CAM_STATE_DIM>();
    // info_factor_matrix_.block<NEW_STATE_DIM, NEW_STATE_DIM>(N, N -
    // NEW_STATE_DIM) = NoiseFactor * state_transition_matrix;
    info_factor_matrix_.block<NEW_STATE_DIM, NEW_STATE_DIM>(OLD_DIM, OLD_DIM) =
        -NoiseFactor;

    // make residual vector
    residual_.segment(0, CURRENT_DIM).setZero();
}
#if 0
	void SquareRootEKFSolver::Propagate(const ImuPreintergration* imu_term)
    {

        static Vector3f gravity(0, GRAVITY, 0);
        Matrix3f R0 = last_state_->Rwi;

        float dt = imu_term->dT * 1e-9;

    	// propagate states
        new_state_->Pwi = last_state_->Pwi + (R0 * imu_term->dP + *vel_ * dt + gravity * (0.5f * dt * dt));
        new_state_->Pw_FEJ = new_state_->Pwi;
        *vel_ += R0 * imu_term->dV + gravity * dt;
        new_state_->Rwi = R0 * imu_term->dR;

        //Make state transition matrix F
        Eigen::Matrix<float,NEW_STATE_DIM,NEW_STATE_DIM> state_transition_matrix;
        state_transition_matrix.setIdentity();
        state_transition_matrix.block<3, 3>(0, 0) = imu_term->dR.transpose();
        state_transition_matrix.block<3, 3>(0, 6) = imu_term->dRdg;
        state_transition_matrix.block<3, 3>(3, 0) = -R0 * crossMat(imu_term->dP);
        state_transition_matrix.block<3, 3>(3, 6) = R0 * imu_term->dPdg;
        state_transition_matrix.block<3, 3>(3, 9) = Matrix3f::Identity() * dt;
        state_transition_matrix.block<3, 3>(9, 0) = -R0 * crossMat(imu_term->dV);
        state_transition_matrix.block<3, 3>(9, 6) = R0 * imu_term->dVdg;
        state_transition_matrix.block<3, 3>(9, 12) = R0 * imu_term->dVda;
        state_transition_matrix.block<3, 3>(3, 12) = R0 * imu_term->dPda;

        //make noise transition matrix
        Eigen::Matrix<float,NEW_STATE_DIM,9> noise_transition_matrix;
        noise_transition_matrix.setZero();
        noise_transition_matrix.block<3, 3>(0, 0).setIdentity();
        noise_transition_matrix.block<3, 3>(3, 6) = R0;
        noise_transition_matrix.block<3, 3>(9, 3) = R0;

        //Make Noise Covariance Matrix Q
        Eigen::Matrix<float,NEW_STATE_DIM,NEW_STATE_DIM> noise_cov = noise_transition_matrix * imu_term->Cov * noise_transition_matrix.transpose();
        noise_cov.block<3, 3>(6, 6) = Matrix3f::Identity() * (Config::GyroBiasNoise2 * dt * dt);
        noise_cov.block<3, 3>(12, 12) = Matrix3f::Identity() * (Config::AccBiasNoise2 * dt * dt);

        Eigen::Matrix<float,NEW_STATE_DIM,NEW_STATE_DIM> NoiseFactor;
        NoiseFactor.setIdentity();
        Eigen::LLT<MatrixXf> chol(noise_cov);
        chol.matrixU().solveInPlace(NoiseFactor);

        //Make information factor matrix
        int N = info_factor_matrix_.rows();
        info_factor_matrix_.conservativeResize(N + NEW_STATE_DIM, N + NEW_STATE_DIM);

    	MatrixXf A = NoiseFactor * state_transition_matrix;

        info_factor_matrix_.topRightCorner(N, NEW_STATE_DIM).setZero();
        info_factor_matrix_.bottomLeftCorner(NEW_STATE_DIM, N).setZero();
        info_factor_matrix_.block<NEW_STATE_DIM, NEW_STATE_DIM>(N, N - NEW_STATE_DIM) = A;
        info_factor_matrix_.bottomRightCorner<NEW_STATE_DIM, NEW_STATE_DIM>() = -NoiseFactor;



        // make residual vector
        residual_ = VectorXf::Zero(info_factor_matrix_.rows());

    }

    void SquareRootEKFSolver::PropagateNew(const ImuPreintergration* imu_term)
    {
        static Vector3f gravity(0, GRAVITY, 0);
        Matrix3f R0 = last_state_->Rwi;

        float dt = imu_term->dT * 1e-9;

    	// propagate states
        new_state_->Pwi = last_state_->Pwi + (R0 * imu_term->dP + *vel_ * dt + gravity * (0.5f * dt * dt));
        new_state_->Pw_FEJ = new_state_->Pwi;
        *vel_ += R0 * imu_term->dV + gravity * dt;
        new_state_->Rwi = R0 * imu_term->dR;

        //Make state transition matrix F
        Eigen::Matrix<float, NEW_STATE_DIM, NEW_STATE_DIM> state_transition_matrix;
        state_transition_matrix.setIdentity();
        state_transition_matrix.block<3, 3>(0, 0) = imu_term->dR.transpose();
        state_transition_matrix.block<3, 3>(0, 6) = imu_term->dRdg;
        state_transition_matrix.block<3, 3>(3, 0) = -R0 * crossMat(imu_term->dP);
        state_transition_matrix.block<3, 3>(3, 6) = R0 * imu_term->dPdg;
        state_transition_matrix.block<3, 3>(3, 9) = Matrix3f::Identity() * dt;
        state_transition_matrix.block<3, 3>(3, 12) = R0 * imu_term->dPda;
        state_transition_matrix.block<3, 3>(9, 0) = -R0 * crossMat(imu_term->dV);
        state_transition_matrix.block<3, 3>(9, 6) = R0 * imu_term->dVdg;
        state_transition_matrix.block<3, 3>(9, 12) = R0 * imu_term->dVda;

        //make noise transition matrix
        Eigen::Matrix<float, NEW_STATE_DIM, 9> noise_transition_matrix;
        noise_transition_matrix.setZero();
        noise_transition_matrix.block<3, 3>(0, 0).setIdentity();
        noise_transition_matrix.block<3, 3>(3, 6) = R0;
        noise_transition_matrix.block<3, 3>(9, 3) = R0;

        //Make Noise Covariance Matrix Q
        Eigen::Matrix<float, NEW_STATE_DIM, NEW_STATE_DIM> noise_cov = noise_transition_matrix * imu_term->Cov * noise_transition_matrix.transpose();
        noise_cov.block<3, 3>(6, 6) = Matrix3f::Identity() * (Config::GyroBiasNoise2 * dt * dt);
        noise_cov.block<3, 3>(12, 12) = Matrix3f::Identity() * (Config::AccBiasNoise2 * dt * dt);

        Eigen::Matrix<float, NEW_STATE_DIM, NEW_STATE_DIM> NoiseFactor;
        NoiseFactor.setIdentity();
        Eigen::LLT<MatrixXf> chol(noise_cov);
        chol.matrixU().solveInPlace(NoiseFactor);

        //Make information factor matrix
        int N = info_factor_matrix_.rows();
        info_factor_matrix_.conservativeResize(N + NEW_STATE_DIM, N + NEW_STATE_DIM);

        info_factor_matrix_.topRightCorner(N, NEW_STATE_DIM).setZero();
        info_factor_matrix_.bottomLeftCorner(NEW_STATE_DIM, N).setZero();
        MatrixXf R = NoiseFactor * state_transition_matrix;
        info_factor_matrix_.block<NEW_STATE_DIM,IMU_STATE_DIM>(N, 0) = R.rightCols<IMU_STATE_DIM>();
        info_factor_matrix_.block<NEW_STATE_DIM, CAM_STATE_DIM>(N, N-CAM_STATE_DIM) = R.leftCols<CAM_STATE_DIM>();
        //info_factor_matrix_.block<NEW_STATE_DIM, NEW_STATE_DIM>(N, N - NEW_STATE_DIM) = NoiseFactor * state_transition_matrix;
        info_factor_matrix_.bottomRightCorner<NEW_STATE_DIM, NEW_STATE_DIM>() = -NoiseFactor;

        // make residual vector
        residual_ = VectorXf::Zero(info_factor_matrix_.rows());


    }

    void SquareRootEKFSolver::marginalize()
    {
        int iDim = 0;
        std::vector<int> v_MarginDIM,v_RemainDIM;
        std::vector<CamState*> v_CamStateNew;
        for (int i=0,n = cam_states_.size();i<n;++i)
        {
            auto state = cam_states_[i];
            if (state->flag_to_marginalize)
                for (int i = 0; i < CAM_STATE_DIM; i++)
                    v_MarginDIM.push_back(iDim++);
            else
            {
                for (int i = 0; i < CAM_STATE_DIM; i++)
                    v_RemainDIM.push_back(iDim++);
                v_CamStateNew.push_back(state);
            }
        }
        cam_states_ = v_CamStateNew;
        cam_states_.push_back(new_state_);
        for (int i=0,n=cam_states_.size();i<n;++i)
        {
            cam_states_[i]->index_in_window = i;
        }

        for (int i = 0; i < IMU_STATE_DIM; i++)
        {
            v_MarginDIM.push_back(iDim++);
        }
        for (int i = 0; i < NEW_STATE_DIM; i++)
        {
            v_RemainDIM.push_back(iDim++);
        }

        int nCols = info_factor_matrix_.cols();

        //rearrange new information factor matrix
        MatrixXf NewInfoFactor(nCols, nCols);
        int index = 0;
        for (auto idx : v_MarginDIM)
        {
            NewInfoFactor.col(index++) = info_factor_matrix_.col(idx);
        }
        for (auto idx : v_RemainDIM)
        {
            NewInfoFactor.col(index++) = info_factor_matrix_.col(idx);
        }

        //use qr to marginalize states
        Eigen::HouseholderQR<MatrixXf> qr(NewInfoFactor);
        int nRemain = v_RemainDIM.size();

        info_factor_matrix_ = qr.matrixQR().bottomRightCorner(nRemain, nRemain).triangularView<Eigen::Upper>();

        m_infoFactorInverseMatrix = MatrixXf::Identity(nRemain, nRemain);
        info_factor_matrix_.triangularView<Eigen::Upper>().solveInPlace(m_infoFactorInverseMatrix);

        residual_.resize(info_factor_matrix_.rows(), 1);
        residual_.setZero();



    }

    void SquareRootEKFSolver::marginalizeNew()
    {
        int iDim = 0;
        std::vector<int> v_MarginDIM, v_RemainDIM;
        std::vector<CamState*> v_CamStateNew;

        int N = info_factor_matrix_.cols();
        for (int i=0;i<IMU_STATE_DIM;++i)
        {
            v_MarginDIM.push_back(i);
            v_RemainDIM.push_back(N - IMU_STATE_DIM + i);
        }
        iDim = IMU_STATE_DIM;
        for (int i = 0, n = cam_states_.size(); i < n; ++i)
        {
            auto state = cam_states_[i];
            if (state->flag_to_marginalize)
                for (int i = 0; i < CAM_STATE_DIM; i++)
                    v_MarginDIM.push_back(iDim++);
            else
            {
                for (int i = 0; i < CAM_STATE_DIM; i++)
                    v_RemainDIM.push_back(iDim++);
                v_CamStateNew.push_back(state);
            }
        }
        cam_states_ = v_CamStateNew;
        cam_states_.push_back(new_state_);
        for (int i = 0, n = cam_states_.size(); i < n; ++i)
        {
            cam_states_[i]->index_in_window = i;
        }

        for (int i = 0; i < CAM_STATE_DIM; i++)
        {
            v_RemainDIM.push_back(iDim++);
        }

        int nCols = info_factor_matrix_.cols();

        //rearrange new information factor matrix
        MatrixXf NewInfoFactor(nCols, nCols);
        int index = 0;
        for (auto idx : v_MarginDIM)
        {
            NewInfoFactor.col(index++) = info_factor_matrix_.col(idx);
        }
        for (auto idx : v_RemainDIM)
        {
            NewInfoFactor.col(index++) = info_factor_matrix_.col(idx);
        }





        //use qr to marginalize states
        Eigen::HouseholderQR<MatrixXf> qr(NewInfoFactor);
        int nRemain = v_RemainDIM.size();

        info_factor_matrix_ = qr.matrixQR().bottomRightCorner(nRemain, nRemain).triangularView<Eigen::Upper>();
    	

        m_infoFactorInverseMatrix = MatrixXf::Identity(nRemain, nRemain);
        info_factor_matrix_.triangularView<Eigen::Upper>().solveInPlace(m_infoFactorInverseMatrix);

        residual_.resize(info_factor_matrix_.rows(), 1);
        residual_.setZero();

    }

#endif

bool SquareRootEKFSolver::MahalanobisTest(PointState* state) {
    VectorXf z = state->H.rightCols<1>();
    // int nExceptPoint = state->H.cols() - 1;
    int num_obs = z.rows();
#if USE_NAIVE_ML_DATAASSOCIATION
    float phi;
    static const float ImageNoise2 =
        SensorConfig::Instance().GetCameraParams(0).image_noise *
        SensorConfig::Instance().GetCameraParams(0).image_noise;
    if (state->flag_slam_point) {
        Matrix2f S;
        Matrix2f R = MatrixXf::Identity(2, 2) * ImageNoise2 * 2;
        MatrixXf E;
        E.resize(CURRENT_DIM, 9);
        int iLeft = IMU_STATE_DIM;
        E.leftCols<3>() = info_factor_matrix_after_mariginal_.block(
            0, state->index_in_window * 3 + iLeft, CURRENT_DIM, 3);
        E.rightCols<6>() = info_factor_matrix_after_mariginal_.block(
            0,
            state->host->last_obs_->link_frame->state->index_in_window *
                    CAM_STATE_DIM +
                iLeft + 3 * slam_point_.size(),
            CURRENT_DIM, 6);
        Matrix9f F = E.transpose() * E;
        S = state->H.leftCols(9) * F.inverse() *
            state->H.leftCols(9).transpose();
        S.noalias() += R;

        phi = z.transpose() * S.inverse() * z;
    } else {
        phi = z.dot(z) / (2 * ImageNoise2);
    }
#else
    MatrixXf S = MatrixXf::Zero(num_obs, num_obs);
    MatrixXf R = MatrixXf::Identity(num_obs, num_obs) * ImageNoise2 * 2;

    MatrixXf B =
        state->H.leftCols(nExceptPoint) *
        m_infoFactorInverseMatrix.topLeftCorner(nExceptPoint, nExceptPoint)
            .triangularView<Eigen::Upper>();
    nn S.noalias() += R;

    float phi = z.transpose() * S.llt().solve(z);
#endif

    assert(num_obs < 80);
#if OUTPUT_DEBUG_INFO
    if (phi < chi2LUT[num_obs])
        printf("#### MahalanobisTest Success\n");
    else
        printf("#### MahalanobisTest Fail\n");

#endif
    return phi < chi2LUT[num_obs];
}

void rowMajorMatrixQRByGivensInMsckf(MatrixHfR& H, int row, int col) {
    (void)row;
    // int nrows = row;
    int nCols = col;
    assert(H.IsRowMajor);
    for (size_t j = 0; j < 3; ++j) {
        for (size_t i = H.rows() - 1; i > j; i--) {
            float* pI = H.row(i).data();
            float* pJ = H.row(i - 1).data();
            float c, s;
            float alpha = pJ[j];
            float beta = pI[j];
            if (fabs(beta) < FLT_EPSILON) {
                continue;
            } else if (fabs(alpha) < FLT_EPSILON) {
                for (int k = j; k < nCols; ++k) {
                    float x = -pI[k];
                    float y = pJ[k];

                    pJ[k] = x;
                    pI[k] = y;
                }
                continue;
            } else if (fabs(beta) > fabs(alpha)) {
                s = 1 / sqrt(1 + pow(alpha / beta, 2));
                c = -alpha / beta * s;
            } else {
                c = 1 / sqrt(1 + pow(beta / alpha, 2));
                s = -beta / alpha * c;
            }
            for (int k = j; k < nCols; ++k) {
                float x = c * pJ[k] - s * pI[k];
                float y = s * pJ[k] + c * pI[k];

                pJ[k] = x;
                pI[k] = y;
            }
        }
    }
}

int SquareRootEKFSolver::ComputeJacobians(Landmark* track) {
    // observation number
    int index = 0;

    CamModel::Ptr cam_model = SensorConfig::Instance().GetCamModel(0);
    static Vector3f Tci = cam_model->getTci();
    int num_cams = cam_states_.size();

    const int CAM_STATE_IDX = 3;

    const int RESIDUAL_IDX = 3 + CAM_STATE_DIM * num_cams;

    int num_obs = track->visual_obs.size();

    if (track->point_state_->flag_slam_point) {
        num_obs = 1;
    }
    // float huberThresh = 500.f;
    float cutOffThresh = 10.f;

    // Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> H;
    auto& H = track->point_state_->H;
    H.setZero(num_obs * 2, 3 + (CAM_STATE_DIM * num_cams) + 1);
    auto calcObsJac = [&](VisualObservation::Ptr ob) {
        int cam_id = ob->link_frame->state->index_in_window;

        Matrix3f Riw = ob->link_frame->state->Rwi.transpose();
        Vector3f Pi =
            Riw * (track->point_state_->Pw - ob->link_frame->state->Pwi);
        Vector3f Pi_FEJ =
            Riw * (track->point_state_->Pw_FEJ - ob->link_frame->state->Pw_FEJ);

        Vector2f px = cam_model->imuToImage(Pi);
        Vector2f r = ob->px - px;
        float reprojErr = r.norm();

        if (reprojErr > cutOffThresh) {
            track->Reproject();
            // track->DrawObservationsAndReprojection(1);
            printf("%f %f ->%f %f\n", ob->px.x(), ob->px.y(), px.x(), px.y());
            return false;
        }

        H.block<2, 1>(2 * index, RESIDUAL_IDX) = r;

        Matrix23f J23;
        cam_model->camToImage(Pi_FEJ + Tci, J23);

        H.block<2, 3>(2 * index, CAM_STATE_IDX + CAM_STATE_DIM * cam_id) =
            J23 * crossMat(Pi_FEJ);
        H.block<2, 3>(2 * index, CAM_STATE_IDX + CAM_STATE_DIM * cam_id + 3) =
            -J23 * Riw;
        H.block<2, 3>(2 * index, 0) = J23 * Riw;

        return true;
    };

    if (track->point_state_->flag_slam_point) {
        if (calcObsJac(track->last_obs_)) {
            index++;
        }
        if (index != num_obs) {
            return 0;
        }
    } else {
        for (auto& ob : track->visual_obs) {
            if (calcObsJac(ob)) {
                index++;
            }
        }
    }

    if (index != num_obs) {
        if (index < 2) return 0;
        num_obs = index;
        H.conservativeResize(2 * index, Eigen::NoChange);
    }

    return 2 * num_obs;
}

int SquareRootEKFSolver::_AddNewSlamPointConstraint() {
    int num_obs = 0;

    for (auto point : slam_point_) {
        int obs = ComputeJacobians(point->host);

        if (obs && MahalanobisTest(point))
            num_obs += obs;
        else
            point->flag_to_marginalize = true;
    }

    if (new_slam_point_.empty()) {
        info_factor_matrix_.topLeftCorner(CURRENT_DIM, CURRENT_DIM) =
            info_factor_matrix_after_mariginal_.topLeftCorner(CURRENT_DIM,
                                                              CURRENT_DIM);
        return num_obs;
    }

    int nNewSlamPointStates = new_slam_point_.size() * 3;

    int nOldSlamPointStates = slam_point_.size() * 3;
    int num_cams = cam_states_.size();

    int nLeft = nOldSlamPointStates + IMU_STATE_DIM;

    int nRight = num_cams * CAM_STATE_DIM;

    int nNewCam = nLeft + nNewSlamPointStates;

    info_factor_matrix_.topLeftCorner(nLeft, nLeft) =
        info_factor_matrix_after_mariginal_.topLeftCorner(nLeft, nLeft);
    info_factor_matrix_.block(0, nNewCam, nLeft, nRight) =
        info_factor_matrix_after_mariginal_.block(0, nLeft, nLeft, nRight);
    info_factor_matrix_.block(nNewCam, 0, nRight, nLeft) =
        info_factor_matrix_after_mariginal_.block(nLeft, 0, nRight, nLeft);
    info_factor_matrix_.block(nNewCam, nNewCam, nRight, nRight) =
        info_factor_matrix_after_mariginal_.block(nLeft, nLeft, nRight, nRight);

    info_factor_matrix_
        .block(0, nLeft, CURRENT_DIM + nNewSlamPointStates, nNewSlamPointStates)
        .setZero();
    info_factor_matrix_
        .block(nLeft, 0, nNewSlamPointStates, CURRENT_DIM + nNewSlamPointStates)
        .setZero();

    CURRENT_DIM += nNewSlamPointStates;

    for (auto state : new_slam_point_) {
        state->host->RemoveUselessObservationForSlamPoint();
    }

    for (auto& state : new_slam_point_) {
        num_obs += state->H.rows();
    }

    slam_point_.insert(slam_point_.end(), new_slam_point_.begin(),
                       new_slam_point_.end());

    for (int i = 0, n = slam_point_.size(); i < n; ++i) {
        slam_point_[i]->index_in_window = i;
    }

    new_slam_point_.clear();

    return num_obs;
}

int SquareRootEKFSolver::AddSlamPointConstraint() {
    int num_obs = 0;
    return num_obs;
}

void SquareRootEKFSolver::AddMsckfPoint(PointState* state) {
    // constexpr int MAX_DIM =
    // 3 + (CAM_STATE_DIM * MAX_WINDOW_SIZE + IMU_STATE_DIM + 1);
    static MatrixHfR H;
    // do null space trick to get pose constraint
    int col = state->H.cols();
    int row = state->H.rows();
    H.topLeftCorner(row, col) = state->H;

    rowMajorMatrixQRByGivensInMsckf(H, row, col);
    state->H = H.block(3, 3, row - 3, col - 3);

    msckf_points_.push_back(state);
}

void SquareRootEKFSolver::AddSlamPoint(PointState* state) {
    new_slam_point_.push_back(state);
    state->flag_slam_point = true;
}

void SquareRootEKFSolver::AddVelocityConstraint(int nRows) {
    static float invSigma = 1.0 / 1e-2;

    static float invRotSigma = 1.0 / 1e-5;
    static float invPosSigma = 1.0 / 1e-2;

    int velIdx = 3;

    stacked_matrix_.middleRows(nRows, 9).setZero();
    Matrix3f dR = new_state_->Rwi.transpose() * last_state_->Rwi;
    Vector3f dP = new_state_->Pwi - last_state_->Pwi;

    Eigen::AngleAxisf angleAxis;
    angleAxis.fromRotationMatrix(dR);
    Vector3f so3 = angleAxis.axis() * angleAxis.angle();
    Matrix3f crossSo3 = crossMat(so3);
    float absAngle = fabs(angleAxis.angle());
    Matrix3f Jr, Jl;
    if (absAngle < FLT_EPSILON) {
        Jr.setIdentity();
        Jl.setIdentity();
    } else {
        Jr = Eigen::Matrix3f::Identity() + 0.5 * crossSo3 +
             (pow(1.f / absAngle, 2) -
              (1 + cos(absAngle) / (2 * absAngle * sin(absAngle)))) *
                 crossSo3 * crossSo3;
        Jl = Jr.transpose();
    }
    stacked_matrix_.block<3, 3>(nRows, CURRENT_DIM - 2 * CAM_STATE_DIM) =
        Jr * invRotSigma;
    stacked_matrix_.block<3, 3>(nRows, CURRENT_DIM - CAM_STATE_DIM) =
        -Jl * invRotSigma;
    obs_residual_.segment(nRows, 3) = -so3 * invRotSigma;
    stacked_matrix_.block<3, 3>(nRows + 3,
                                CURRENT_DIM - 2 * CAM_STATE_DIM + 3) =
        -Matrix3f::Identity() * invPosSigma;
    stacked_matrix_.block<3, 3>(nRows + 3, CURRENT_DIM - CAM_STATE_DIM + 3) =
        Matrix3f::Identity() * invPosSigma;
    obs_residual_.segment(nRows + 3, 3) = -dP * invPosSigma;
    stacked_matrix_.block<3, 3>(nRows + 6, velIdx) =
        Matrix3f::Identity() * invSigma;
    obs_residual_.segment(nRows + 6, 3) = -*vel_ * invSigma;
    stacked_rows_ += 9;
}

int SquareRootEKFSolver::StackInformationFactorMatrix() {
    int nTotalObs = 0;
    int nCamStartIdx = IMU_STATE_DIM;
    int nVisualObs = 0;
    int rffIdx = 0;

    nVisualObs += _AddNewSlamPointConstraint();
    nCamStartIdx += slam_point_.size() * 3;

    int nOldStates = CURRENT_DIM;
    int nCamStates = cam_states_.size() * CAM_STATE_DIM;

    using namespace std;
    for (auto& track : msckf_points_) {
        nVisualObs += track->H.rows();
    }

    nTotalObs += nVisualObs;

    stacked_matrix_.topLeftCorner(nTotalObs, nOldStates).setZero();
    obs_residual_.segment(0, nTotalObs).setZero();

    if (!nVisualObs) {
        stacked_rows_ = 0;
        return 0;
    }

    float invSigma =
        1 / SensorConfig::Instance().GetCameraParams(0).image_noise;
    MatrixXf H_j;

    for (auto& track : msckf_points_) {
        int num_obs = track->H.rows();
        track->H *= invSigma;

        stacked_matrix_.block(rffIdx, nCamStartIdx, num_obs, nCamStates) =
            track->H.leftCols(nCamStates);
        //.triangularView<Eigen::Upper>();

        obs_residual_.segment(rffIdx, num_obs) = track->H.rightCols<1>();
        rffIdx += num_obs;
    }

    int slamPointIdx = IMU_STATE_DIM;

    for (auto& track : slam_point_) {
        if (track->flag_to_marginalize) continue;
        int num_obs = track->H.rows();
        track->H *= invSigma;
        int slamId = track->index_in_window;

        stacked_matrix_.block(rffIdx, nCamStartIdx, num_obs, nCamStates) =
            track->H.middleCols(3, nCamStates);
        stacked_matrix_.block(rffIdx, slamPointIdx + slamId * 3, num_obs, 3) =
            track->H.leftCols<3>();
        obs_residual_.segment(rffIdx, num_obs) = track->H.rightCols<1>();
        rffIdx += num_obs;
    }

    stacked_rows_ = nTotalObs;
    return nTotalObs;
}

void SquareRootEKFSolver::SolveAndUpdateStates() {
    int haveNewInformation = stacked_rows_;
    if (haveNewInformation) {
        TickTock::Start("Givens");
        _UpdateByGivensRotations(haveNewInformation, CURRENT_DIM + 1);
        TickTock::Stop("Givens");

        TickTock::Start("Inverse");

        VectorXf dx =
            info_factor_matrix_.topLeftCorner(CURRENT_DIM, CURRENT_DIM)
                .triangularView<Eigen::Upper>()
                .solve(residual_.segment(0, CURRENT_DIM));
        TickTock::Stop("Inverse");

        int iDim = 0;

        auto& imuBuffer = ImuBuffer::Instance();
        imuBuffer.UpdateBias(dx.segment<3>(iDim), dx.segment<3>(iDim + 6));
        *vel_ += dx.segment<3>(iDim + 3);
        iDim += IMU_STATE_DIM;

        for (auto& pointState : slam_point_) {
            pointState->Pw += dx.segment<3>(iDim);
            iDim += 3;
        }

        for (auto& camState : cam_states_) {
            camState->Rwi =
                camState->Rwi *
                Sophus::SO3Group<float>::exp(dx.segment<3>(iDim)).matrix();
            iDim += 3;
            camState->Pwi += dx.segment<3>(iDim);
            iDim += 3;
        }
    }

    msckf_points_.clear();
}

#ifdef PLATFORM_ARM
void SquareRootEKFSolver::_updateByGivensRotationsNeon() {}

#endif

void SquareRootEKFSolver::_UpdateByGivensRotations(int row, int col) {
    for (int j = 0; j < col - 1; ++j) {
        for (int i = row - 1; i >= 0; --i) {
            if (i == 0) {
                float* pI = stacked_matrix_.row(i).data();
                float alpha = info_factor_matrix_(j, j);
                float beta = pI[j];
                float c, s;
                if (fabs(beta) < FLT_EPSILON) {
                    continue;
                } else if (fabs(beta) > fabs(alpha)) {
                    s = 1 / sqrt(1 + pow(alpha / beta, 2));
                    c = -alpha / beta * s;
                } else {
                    c = 1 / sqrt(1 + pow(beta / alpha, 2));
                    s = -beta / alpha * c;
                }
                for (int k = j; k < col; ++k) {
                    if (k == col - 1) {
                        float x = residual_(j);
                        float y = obs_residual_(i);
                        obs_residual_(i) = s * x + c * y;
                        residual_(j) = c * x + -s * y;

                    } else {
                        float x = info_factor_matrix_(j, k);
                        float y = pI[k];
                        info_factor_matrix_(j, k) = c * x + -s * y;
                        pI[k] = s * x + c * y;
                    }
                }
            } else {
                float* pJ = stacked_matrix_.row(i - 1).data();
                float* pI = stacked_matrix_.row(i).data();
                float alpha = pJ[j];
                float beta = pI[j];
                float c, s;
                if (fabs(beta) < FLT_EPSILON) {
                    continue;
                } else if (fabs(beta) > fabs(alpha)) {
                    s = 1 / sqrt(1 + pow(alpha / beta, 2));
                    c = -alpha / beta * s;
                } else {
                    c = 1 / sqrt(1 + pow(beta / alpha, 2));
                    s = -beta / alpha * c;
                }
                int k;
                for (k = j; k < col - 1; ++k) {
                    float x = pJ[k];
                    float y = pI[k];
                    pJ[k] = c * x + -s * y;
                    pI[k] = s * x + c * y;
                }
                float x = obs_residual_[i - 1];
                float y = obs_residual_[i];
                obs_residual_[i - 1] = c * x + -s * y;
                obs_residual_[i] = s * x + c * y;
            }
        }
    }
}

int SquareRootEKFSolver::_AddPositionContraint(int nRows) {
    static float invPosSigma = 1.0 / 0.05;

    stacked_matrix_.middleRows(nRows, 3).setZero();

    Vector3f dP = new_state_->Pwi - last_state_->Pwi;

    stacked_matrix_.block<3, 3>(nRows + 3,
                                CURRENT_DIM - 2 * CAM_STATE_DIM + 3) =
        -Matrix3f::Identity() * invPosSigma;
    stacked_matrix_.block<3, 3>(nRows + 3, CURRENT_DIM - CAM_STATE_DIM + 3) =
        Matrix3f::Identity() * invPosSigma;
    obs_residual_.segment(nRows + 3, 3) = -dP * invPosSigma;
    return 3;
}

void SquareRootEKFSolver::_MarginByGivensRotation() {
    int nRows = CURRENT_DIM;
    for (int j = 0; j < CURRENT_DIM; ++j) {
        for (int i = nRows - 1; i > j; --i) {
            float* pJ = info_factor_matrix_to_marginal_.row(i - 1).data();
            float* pI = info_factor_matrix_to_marginal_.row(i).data();
            float alpha = pJ[j];
            float beta = pI[j];
            float c, s;
            if (fabs(beta) < FLT_EPSILON) {
                continue;
            } else if (fabs(alpha) < FLT_EPSILON) {
                for (int k = j; k < CURRENT_DIM; ++k) {
                    std::swap(pJ[k], pI[k]);
                }
                continue;
                ;
            } else if (fabs(beta) > fabs(alpha)) {
                s = 1 / sqrt(1 + pow(alpha / beta, 2));
                c = -alpha / beta * s;
            } else {
                c = 1 / sqrt(1 + pow(beta / alpha, 2));
                s = -beta / alpha * c;
            }
            for (int k = j; k < CURRENT_DIM; ++k) {
                float x = pJ[k];
                float y = pI[k];
                pJ[k] = c * x - s * y;
                pI[k] = s * x + c * y;
            }
        }
    }
}

void SquareRootEKFSolver::MarginalizeGivens() {
    int iDim = 0;
    std::vector<int> v_MarginDIM, v_RemainDIM;
    std::vector<CamState*> v_CamStateNew;
    std::vector<PointState*> v_PointStateNew;

    for (int i = 0; i < IMU_STATE_DIM; ++i) {
        v_MarginDIM.push_back(i);
        v_RemainDIM.push_back(CURRENT_DIM - IMU_STATE_DIM + i);
    }
    iDim = IMU_STATE_DIM;

    for (int i = 0, n = slam_point_.size(); i < n; ++i) {
        if (slam_point_[i]->flag_to_marginalize ||
            slam_point_[i]->flag_to_next_marginalize)
            slam_point_[i]->host->flag_dead = true;
        if (slam_point_[i]->host->flag_dead) {
            for (int j = 0; j < 3; ++j) {
                v_MarginDIM.push_back(iDim++);
            }
        } else {
            for (int j = 0; j < 3; ++j) {
                v_RemainDIM.push_back(iDim++);
            }
            v_PointStateNew.push_back(slam_point_[i]);
        }
    }

    slam_point_ = v_PointStateNew;
    for (int i = 0, n = slam_point_.size(); i < n; ++i) {
        slam_point_[i]->index_in_window = i;
    }

    for (int i = 0, n = cam_states_.size(); i < n; ++i) {
        auto state = cam_states_[i];
        if (state->flag_to_marginalize)
            for (int i = 0; i < CAM_STATE_DIM; i++)
                v_MarginDIM.push_back(iDim++);
        else {
            for (int i = 0; i < CAM_STATE_DIM; i++)
                v_RemainDIM.push_back(iDim++);
            v_CamStateNew.push_back(state);
        }
    }

    cam_states_ = v_CamStateNew;
    cam_states_.push_back(new_state_);
    for (int i = 0, n = cam_states_.size(); i < n; ++i) {
        cam_states_[i]->index_in_window = i;
    }

    for (int i = 0; i < CAM_STATE_DIM; i++) {
        v_RemainDIM.push_back(iDim++);
    }

    // rearrange new information factor matrix

    int index = 0;
    for (auto idx : v_MarginDIM) {
        info_factor_matrix_to_marginal_.col(index++).segment(0, CURRENT_DIM) =
            info_factor_matrix_.col(idx).segment(0, CURRENT_DIM);
    }
    for (auto idx : v_RemainDIM) {
        info_factor_matrix_to_marginal_.col(index++).segment(0, CURRENT_DIM) =
            info_factor_matrix_.col(idx).segment(0, CURRENT_DIM);
    }

    // use qr to marginalize states

    _MarginByGivensRotation();

    int OLD_DIM = CURRENT_DIM;
    CURRENT_DIM = v_RemainDIM.size();

    info_factor_matrix_after_mariginal_.topLeftCorner(CURRENT_DIM,
                                                      CURRENT_DIM) =
        info_factor_matrix_to_marginal_.block(OLD_DIM - CURRENT_DIM,
                                              OLD_DIM - CURRENT_DIM,
                                              CURRENT_DIM, CURRENT_DIM);
}

void SquareRootEKFSolver::MarginalizeStatic() {
    int iDim = 0;
    std::vector<int> v_MarginDIM, v_RemainDIM;
    std::vector<CamState*> v_CamStateNew;

    for (int i = 0; i < IMU_STATE_DIM; ++i) {
        v_MarginDIM.push_back(i);
        v_RemainDIM.push_back(CURRENT_DIM - IMU_STATE_DIM + i);
    }
    iDim = IMU_STATE_DIM;
    for (int i = 0, n = cam_states_.size(); i < n; ++i) {
        auto state = cam_states_[i];
        if (state->flag_to_marginalize)
            for (int i = 0; i < CAM_STATE_DIM; i++)
                v_MarginDIM.push_back(iDim++);
        else {
            for (int i = 0; i < CAM_STATE_DIM; i++)
                v_RemainDIM.push_back(iDim++);
            v_CamStateNew.push_back(state);
        }
    }
    cam_states_ = v_CamStateNew;
    cam_states_.push_back(new_state_);
    for (int i = 0, n = cam_states_.size(); i < n; ++i) {
        cam_states_[i]->index_in_window = i;
    }

    for (int i = 0; i < CAM_STATE_DIM; i++) {
        v_RemainDIM.push_back(iDim++);
    }

    // rearrange new information factor matrix

    int index = 0;
    for (auto idx : v_MarginDIM) {
        info_factor_matrix_to_marginal_.col(index++) =
            info_factor_matrix_.col(idx);
    }
    for (auto idx : v_RemainDIM) {
        info_factor_matrix_to_marginal_.col(index++) =
            info_factor_matrix_.col(idx);
    }

    // use qr to marginalize states

    Eigen::HouseholderQR<MatrixXf> qr(
        info_factor_matrix_to_marginal_.topLeftCorner(CURRENT_DIM,
                                                      CURRENT_DIM));
    CURRENT_DIM = v_RemainDIM.size();

    info_factor_matrix_.topLeftCorner(CURRENT_DIM, CURRENT_DIM) =
        qr.matrixQR()
            .bottomRightCorner(CURRENT_DIM, CURRENT_DIM)
            .triangularView<Eigen::Upper>();

    // m_infoFactorInverseMatrix = MatrixXf::Identity(CURRENT_DIM, CURRENT_DIM);
    // info_factor_matrix_.topLeftCorner(CURRENT_DIM,
    // CURRENT_DIM).triangularView<Eigen::Upper>().solveInPlace(m_infoFactorInverseMatrix);

    residual_.segment(0, CURRENT_DIM).setZero();
}

}  // namespace DeltaVins
