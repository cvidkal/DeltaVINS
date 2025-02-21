#include "Algorithm/VIOAlgorithm.h"

#include <Eigen/Core>

#include "Algorithm/DataAssociation/DataAssociation.h"
#include "Algorithm/vision/FeatureTrackerOpticalFlow_Chen.h"
#include "Algorithm/vision/camModel/camModel.h"
#include "IO/dataBuffer/imuBuffer.h"
#include "precompile.h"
#include "utils/TickTock.h"
#include "utils/constantDefine.h"
#include "utils/utils.h"
#include "utils/SensorConfig.h"

namespace DeltaVins {
VIOAlgorithm::VIOAlgorithm() {
    feature_tracker_ = new FeatureTrackerOpticalFlow_Chen(Config::MaxNumToTrack,
                                                          Config::MaskSize);
    solver_ = new SquareRootEKFSolver();
    DataAssociation::InitDataAssociation(solver_);
    states_.init_state_ = InitState::NeedFirstFrame;
}

VIOAlgorithm::~VIOAlgorithm() {
    if (feature_tracker_) {
        delete feature_tracker_;
        feature_tracker_ = nullptr;
    }
    if (solver_) {
        delete solver_;
        solver_ = nullptr;
    }
}

void VIOAlgorithm::_TrackFrame(const ImageData::Ptr imageData) {
    // Track Feature
    feature_tracker_->MatchNewFrame(states_.tfs_, imageData, frame_now_.get());
}

void VIOAlgorithm::_Initialization(const ImageData::Ptr imageData) {
    if (!static_initializer_.Initialize(imageData)) {
        LOGI("Initializating...");
        return;
    }

    auto& imuBuffer = ImuBuffer::Instance();
    auto timestamp = imageData->timestamp;
    // Get Gravity
    Vector3f g = imuBuffer.GetGravity(timestamp);
    float g_norm = g.norm();
    LOGI("Gravity:%f %f %f, Gravity norm:%f", g.x(), g.y(), g.z(), g_norm);
    if (std::fabs(g_norm - GRAVITY) > 1.0) {
        LOGW(
            "Gravity is invalid, wait for a stationary state for "
            "initialization !");
        return;
    }
    Matrix3f R = GetRotByAlignVector(g, Eigen::Vector3f(0, 0, -1));
    InitializeStates(R);
    imuBuffer.UpdateBiasByStatic(timestamp);
    _TrackFrame(imageData);
    states_.init_state_ = InitState::FirstFrame;
}

void VIOAlgorithm::AddNewFrame(const ImageData::Ptr imageData, Pose::Ptr pose) {
    TickTock::Start("AddFrame");
    // Process input data
    _PreProcess(imageData);

    if (states_.init_state_ == InitState::FirstFrame) {
        states_.init_state_ = InitState::Initialized;
        return;
    }
    if (states_.init_state_ != InitState::Initialized) {
        return;
    }

#if TEST_VISION_MODULE

    _TestVisionModule(imageData, pose);

#else
    TickTock::Start("Propagate");
    // Propagate states
    _AddImuInformation();

    TickTock::Stop("Propagate");

    TickTock::Start("TrackFeature");

    _TrackFrame(imageData);

    TickTock::Stop("TrackFeature");

    TickTock::Start("Update");

    _SelectKeyframe();

    // Update vision measurement
    _AddMeasurement();
    TickTock::Stop("Update");

    TickTock::Stop("AddFrame");

    // Process output data
    _PostProcess(imageData, pose);
#endif
}
void VIOAlgorithm::SetWorldPointAdapter(WorldPointAdapter* adapter) {
    world_point_adapter_ = adapter;
}

void VIOAlgorithm::SetFrameAdapter(FrameAdapter* adapter) {
    frame_adapter_ = adapter;
}

void VIOAlgorithm::_PreProcess(const ImageData::Ptr imageData) {
    auto timestamp = imageData->timestamp;

    frame_now_ = std::make_shared<Frame>(imageData->sensor_id);
    frame_now_->timestamp = timestamp;

#if ENABLE_VISUALIZER && !defined(PLATFORM_ARM)
    frame_now_->image = imageData->image.clone();  // Only used for debugging
#endif
    static auto& imuBuffer = ImuBuffer::Instance();
    if (states_.init_state_ == InitState::NeedFirstFrame) {
        preintergration_.t0 = timestamp;
        states_.init_state_ = InitState::NotInitialized;
        Matrix3f R = Matrix3f::Identity();
        imuBuffer.SetZeroBias();
        InitializeStates(R);
        return;
    }

    preintergration_.t1 = timestamp;
    imuBuffer.ImuPreIntegration(preintergration_);
    preintergration_.t0 = preintergration_.t1;

    // static FILE* file = fopen("TestResults/preintergration.csv", "w");
    // fprintf(file, "%lld %lld %lld\n", imageData->timestamp,
    // imuBuffer.GetOldestImuData().timestamp,imuBuffer.GetLastImuData().timestamp);

    // Init system
    if (states_.init_state_ == InitState::NotInitialized) {
        _Initialization(imageData);
        return;
    }
}

void VIOAlgorithm::_PostProcess(ImageData::Ptr data, Pose::Ptr pose) {
    (void)data;
    Vector3f Pwi, Vwi;
    Vector3f bg, ba;
    auto* camState = frame_now_->state;
    Matrix3f Rwi = camState->Rwi;

    Pwi = camState->Pwi;
    Vwi = states_.vel;
    pose->timestamp = frame_now_->timestamp;

    pose->Pwb = Pwi * 1e3;
    pose->Rwb = Rwi;

    ImuBuffer::Instance().GetBias(bg, ba);

    Quaternionf _q(Rwi);
    auto camModel = SensorConfig::Instance().GetCamModel(0);
    Eigen::Isometry3f Tci_isometry = Eigen::Isometry3f::Identity();
    Tci_isometry.linear() = camModel->getRci();
    // Tci_isometry.translation() = camModel->getTci();
    Eigen::Isometry3f Twc_isometry = Eigen::Isometry3f::Identity();
    Twc_isometry.linear() = Rwi;
    Twc_isometry.translation() = Pwi;
    Eigen::Isometry3f Twi_isometry = Twc_isometry * Tci_isometry;
    Rwi = Twi_isometry.rotation();
    Pwi = Twi_isometry.translation();
    _q = Quaternionf(Rwi);

    std::string outputName = Config::outputFileName;
    static FILE* file = fopen(outputName.c_str(), "w");
    // static FILE* stdvar = fopen("stdvar.csv", "w");

    // Vector3f ea = Rwi.transpose().eulerAngles(0, 1, 2);
#ifndef PLATORM_ARM
    if (Config::OutputFormat == ResultOutputFormat::EUROC) {
        fprintf(
            file,
            "%lld,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%9.6f,%9.6f,%9.6f,%9.6f,%9.6f,%"
            "9.6f\n",
            pose->timestamp, Pwi[0], Pwi[1], Pwi[2], _q.w(), _q.x(), _q.y(),
            _q.z(), Vwi[0], Vwi[1], Vwi[2], bg[0], bg[1], bg[2], ba[0], ba[1],
            ba[2]);
        fflush(file);
    } else if (Config::OutputFormat == ResultOutputFormat::KITTI) {
        // TODO: Implement KITTI format
        throw std::runtime_error("KITTI format is not implemented");
    } else if (Config::OutputFormat == ResultOutputFormat::TUM) {
        fprintf(file, "%lf %f %f %f %f %f %f %f\n", pose->timestamp / 1e9,
                Pwi[0], Pwi[1], Pwi[2], _q.x(), _q.y(), _q.z(), _q.w());
        fflush(file);
    }
#endif
    if (!Config::NoDebugOutput) {
        LOGI(
            "\nTimestamp:%lld\n Position:%f,%f,%f\n "
            "Q:%f,%f,%f,%f\nVelocity:%f,%f,%f",
            pose->timestamp, Pwi[0], Pwi[1], Pwi[2], _q.w(), _q.x(), _q.y(),
            _q.z(), Vwi[0], Vwi[1], Vwi[2]);
        LOGI("\nGyro Bias:%9.6f,%9.6f,%9.6f\nAcc Bias:%9.6f,%9.6f,%9.6f", bg[0],
             bg[1], bg[2], ba[0], ba[1], ba[2]);
    }
    // fflush(file);

#if ENABLE_VISUALIZER || ENABLE_VISUALIZER_TCP || USE_ROS2
    if (!Config::NoGUI) {
        cv::Mat trackImage;
        _DrawTrackImage(data, trackImage);
        if (frame_adapter_) {
            frame_adapter_->PushImageTexture(trackImage.data, trackImage.cols,
                                             trackImage.rows,
                                             trackImage.channels());
            frame_adapter_->FinishFrame();
        } else {
            LOGW("frame_adapter_ is nullptr");
        }

        // cv::imshow("track", trackImage);
        // imshow("FeatureTrack",trackImage);
        // waitKey(1);
    }
#endif
    TickTock::outputResultConsole();
}

void VIOAlgorithm::_UpdatePointsAndCamsToVisualizer() {
#if ENABLE_VISUALIZER || ENABLE_VISUALIZER_TCP || USE_ROS2

    static std::vector<WorldPointGL> vPointsGL;
    static std::vector<FrameGL> vFramesGL;
    vPointsGL.clear();
    vFramesGL.clear();
    vPointsGL.reserve(300);
    vFramesGL.reserve(100);

    static int visCounter = 0;

    for (auto lTrack : states_.tfs_) {
        if (lTrack->point_state_ && lTrack->point_state_->flag_slam_point) {
            if (lTrack->point_state_->m_idVis < 0)
                lTrack->point_state_->m_idVis = visCounter++;
            vPointsGL.emplace_back(lTrack->point_state_->Pw * 1e3,
                                   lTrack->point_state_->m_idVis);
        }
    }

    for (auto frame : states_.frames_) {
        vFramesGL.emplace_back(frame->state->Rwi.matrix(),
                               frame->state->Pwi * 1e3, frame->state->m_id);
    }
    if (!Config::NoGUI) {
        assert(frame_adapter_ && world_point_adapter_);
        frame_adapter_->PushViewMatrix(vFramesGL);
        world_point_adapter_->PushWorldPoint(vPointsGL);
    }
#endif
}

void VIOAlgorithm::_DrawTrackImage(ImageData::Ptr dataPtr,
                                   cv::Mat& trackImage) {
    cvtColor(dataPtr->image, trackImage, cv::COLOR_GRAY2BGR);

    for (auto lTrack : states_.tfs_) {
        if (!lTrack->flag_dead) {
            if (lTrack->point_state_ && lTrack->point_state_->flag_slam_point)
                lTrack->DrawFeatureTrack(trackImage, _GREEN_SCALAR);
            else if (lTrack->num_obs > 5)
                lTrack->DrawFeatureTrack(trackImage, _BLUE_SCALAR);
            else
                lTrack->DrawFeatureTrack(trackImage, _RED_SCALAR);
        }
    }
}

void VIOAlgorithm::_DrawPredictImage(ImageData::Ptr dataPtr,
                                     cv::Mat& predictImage) {
    static std::ofstream fout("Predict.txt");
    cvtColor(dataPtr->image, predictImage, cv::COLOR_GRAY2BGR);

    for (auto lTrack : states_.tfs_) {
        if (!lTrack->flag_dead) {
            if (lTrack->visual_obs.size() >= 2) {
                // int nSize = lTrack->visual_obs.size();
                cv::line(predictImage,
                         cv::Point(lTrack->last_last_obs_->px.x(),
                                   lTrack->last_last_obs_->px.y()),
                         cv::Point(lTrack->last_obs_->px.x(),
                                   lTrack->last_obs_->px.y()),
                         _GREEN_SCALAR);
                cv::line(predictImage,
                         cv::Point(lTrack->last_last_obs_->px.x(),
                                   lTrack->last_last_obs_->px.y()),
                         cv::Point(lTrack->predicted_px.x(),
                                   lTrack->predicted_px.y()),
                         _BLUE_SCALAR);
                cv::circle(predictImage,
                           cv::Point(lTrack->last_last_obs_->px.x(),
                                     lTrack->last_last_obs_->px.y()),
                           2, _RED_SCALAR);
                cv::circle(predictImage,
                           cv::Point(lTrack->last_obs_->px.x(),
                                     lTrack->last_obs_->px.y()),
                           2, _GREEN_SCALAR);
                cv::circle(predictImage,
                           cv::Point(lTrack->predicted_px.x(),
                                     lTrack->predicted_px.y()),
                           2, _BLUE_SCALAR);

                fout << lTrack->last_obs_->px.x() -
                            lTrack->last_last_obs_->px.x()
                     << " "
                     << lTrack->last_obs_->px.y() -
                            lTrack->last_last_obs_->px.y()
                     << " "
                     << lTrack->predicted_px.x() -
                            lTrack->last_last_obs_->px.x()
                     << " "
                     << lTrack->predicted_px.y() -
                            lTrack->last_last_obs_->px.y()
                     << std::endl;
            }
        }
    }
}

void VIOAlgorithm::InitializeStates(const Matrix3f& Rwi) {
    auto* camState = frame_now_->state;

    camState->Rwi = Rwi;
    camState->Pwi.setZero();
    camState->Pw_FEJ.setZero();
    camState->index_in_window = 0;
    states_.frames_.clear();
    states_.frames_.push_back(frame_now_);
    states_.tfs_.clear();
    states_.vel.setZero();
    states_.static_ = false;
    solver_->Init(camState, &states_.vel, &states_.static_);
}

void VIOAlgorithm::_AddImuInformation() {
    solver_->AddCamState(frame_now_->state);

    solver_->PropagateStatic(&preintergration_);
#if 0
		solver_->PropagateNew(&preintergration_);
#endif
}

void VIOAlgorithm::_RemoveDeadFeatures() {
    states_.tfs_.remove_if([](const Landmark::Ptr& tracked_feature) {
        return tracked_feature->flag_dead;
    });
}

void VIOAlgorithm::_AddMeasurement() {
    _DetectStill();

    TickTock::Start("Margin");
    _MarginFrames();
    TickTock::Stop("Margin");

    if (states_.tfs_.empty()) return;

    TickTock::Start("DataAssociation");
    DataAssociation::DoDataAssociation(states_.tfs_, states_.static_);

    TickTock::Stop("DataAssociation");
#if ENABLE_VISUALIZER && !defined(PLATFORM_ARM)
    DataAssociation::DrawPointsBeforeUpdates(solver_->slam_point_);
#endif

    TickTock::Start("Stack");
    _StackInformationFactorMatrix();

    TickTock::Stop("Stack");

    TickTock::Start("Solve");
    solver_->SolveAndUpdateStates();
    TickTock::Stop("Solve");
#if ENABLE_VISUALIZER && !defined(PLATFORM_ARM)
    DataAssociation::DrawPointsAfterUpdates(solver_->slam_point_);
    if (!Config::NoGUI) cv::waitKey(5);
#endif
#if ENABLE_VISUALIZER_TCP || ENABLE_VISUALIZER || USE_ROS2
    if (!Config::NoGUI) _UpdatePointsAndCamsToVisualizer();
#endif
    _RemoveDeadFeatures();
}

void VIOAlgorithm::_SelectFrames2Margin() {
    int nCams = states_.frames_.size();
    int cnt = 0;
    int nKF = 0;
    for (auto& frame : states_.frames_) {
        if (frame->flag_keyframe) {
            nKF++;
        }

        if (frame->valid_landmark_num == 0) {
            cnt++;
            frame->RemoveAllObservations();
            frame->state->flag_to_marginalize = true;
        }
    }
    if (!cnt && nCams >= MAX_WINDOW_SIZE) {
        static int camIdxToMargin = 0;
        camIdxToMargin += CAM_DELETE_STEP;
        if (camIdxToMargin >= nCams - 1) camIdxToMargin = 1;
        if (nKF > 4) {
            for (int i = 0, j = 0; i < nCams; ++i) {
                if (states_.frames_[i]->flag_keyframe) {
                    if (j) {
                        camIdxToMargin = i;
                        break;
                    }
                    j++;
                }
            }
        } else {
            while (camIdxToMargin < nCams &&
                   states_.frames_[camIdxToMargin]->flag_keyframe) {
                camIdxToMargin++;
            }
            if (camIdxToMargin >= nCams - 1) camIdxToMargin = 1;
        }
        states_.frames_[camIdxToMargin]->RemoveAllObservations();
        states_.frames_[camIdxToMargin]->state->flag_to_marginalize = true;
    }
}

void VIOAlgorithm::_SelectKeyframe() {
    if (states_.tfs_.empty()) return;

    auto setkeyframe = [&]() {
        last_keyframe_->flag_keyframe = true;
        for (auto& point : states_.tfs_) {
            if (!point->host_frame) point->host_frame = last_keyframe_.get();
        }
    };

    if (last_keyframe_ == nullptr) {
        last_keyframe_ = frame_now_;
        setkeyframe();
        return;
    }

    float nLastKeyframePoints = last_keyframe_->valid_landmark_num;

    float nPointsNow = frame_now_->valid_landmark_num;

    if (nPointsNow == 0) {
        last_keyframe_ = nullptr;
        return;
    }

    if (nLastKeyframePoints / nPointsNow < 0.6) {
        last_keyframe_ = frame_now_;
        setkeyframe();
    }
}

void VIOAlgorithm::_MarginFrames() {
    std::vector<Frame::Ptr> vCamStatesNew;

    _SelectFrames2Margin();

    solver_->MarginalizeGivens();

    for (auto frame : states_.frames_)
        if (!frame->state->flag_to_marginalize) vCamStatesNew.push_back(frame);
    states_.frames_ = vCamStatesNew;
    states_.frames_.push_back(frame_now_);
}

void VIOAlgorithm::_StackInformationFactorMatrix() {
    int nDIM = solver_->StackInformationFactorMatrix();
    if (!nDIM) {
        if (states_.static_) {
            solver_->AddVelocityConstraint(nDIM);
        }
    }
}

// TODO: moved to tracker
bool VIOAlgorithm::_VisionStatic() {
    return feature_tracker_->IsStaticLastFrame();
}

void VIOAlgorithm::_DetectStill() {
    static auto& buffer = ImuBuffer::Instance();
    bool bStatic = buffer.DetectStatic(frame_now_->timestamp);

    if (bStatic) {
        if (_VisionStatic()) {
            states_.static_ = true;
            return;
        }
    }

    states_.static_ = false;
}

void VIOAlgorithm::_TestVisionModule(const ImageData::Ptr data,
                                     Pose::Ptr pose) {
    _AddImuInformation();

    _TrackFrame(data);

    _MarginFrames();

    _RemoveDeadFeatures();

    // static FILE* file = fopen("TestResults/tracked_features.csv", "w");
    // fprintf(file, "%lld %zu\n", data->timestamp, states_.tfs_.size());

    LOGI("%zu Point remain", states_.tfs_.size());

#if ENABLE_VISUALIZER
    cv::Mat trackImage;
    cv::Mat PredictImage;
    _DrawTrackImage(data, trackImage);
    _DrawPredictImage(data, PredictImage);
    cv::imshow("Predict", PredictImage);
    cv::waitKey(0);
#elif USE_ROS2
    if (!Config::NoGUI) {
        cv::Mat trackImage;
        _DrawTrackImage(data, trackImage);
        frame_adapter_->PushImageTexture(trackImage.data, trackImage.cols,
                                         trackImage.rows,
                                         trackImage.channels());
    }
#endif
    _PostProcess(data, pose);
}

}  // namespace DeltaVins
