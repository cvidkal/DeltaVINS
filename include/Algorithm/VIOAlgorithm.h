#pragma once

#include <FrameAdapter.h>
#include <WorldPointAdapter.h>

#include "IMU/ImuPreintergration.h"
#include "dataStructure/IO_Structures.h"
#include "solver/SquareRootEKFSolver.h"
#include "vision/FeatureTrackerOpticalFlow.h"
#include "vision/FeatureTrackerOpticalFlow_Chen.h"
#include "Initializer/StaticInitializer.h"
namespace DeltaVins {
class VIOAlgorithm {
   public:
    VIOAlgorithm();
    ~VIOAlgorithm();

    void AddNewFrame(const ImageData::Ptr imageData, Pose::Ptr pose);

    void SetWorldPointAdapter(WorldPointAdapter* adapter);
    void SetFrameAdapter(FrameAdapter* adapter);

   private:
    enum class InitState {
        NeedFirstFrame,
        NotInitialized,
        FirstFrame,
        Initialized,
        Failed
    };
    struct SystemStates {
        Vector3f vel;
        std::vector<Frame::Ptr> frames_;
        std::list<Landmark::Ptr> tfs_;
        bool static_;
        InitState init_state_;
    };

    void _Initialization(const ImageData::Ptr imageData);

    void _PreProcess(const ImageData::Ptr imageData);
    void _PostProcess(ImageData::Ptr data, Pose::Ptr pose);
    void _UpdatePointsAndCamsToVisualizer();
    void _DrawTrackImage(ImageData::Ptr dataPtr, cv::Mat& trackImage);
    void _DrawPredictImage(ImageData::Ptr dataPtr, cv::Mat& predictImage);
    void _TrackFrame(const ImageData::Ptr imageData);
    void InitializeStates(const Matrix3f& Rwi);

    void _AddImuInformation();
    void _RemoveDeadFeatures();
    void _MarginFrames();
    void _StackInformationFactorMatrix();
    void _DetectStill();
    void _TestVisionModule(const ImageData::Ptr data, Pose::Ptr pose);
    void _AddMeasurement();
    void _SelectFrames2Margin();

    bool _VisionStatic();

    FeatureTrackerOpticalFlow_Chen* feature_tracker_ = nullptr;
    SquareRootEKFSolver* solver_ = nullptr;
    SystemStates states_;
    Frame::Ptr frame_now_ = nullptr;

    ImuPreintergration preintergration_;

    bool initialized_;

    Frame::Ptr last_keyframe_ = nullptr;
    void _SelectKeyframe();

    /************* Output **********************/

    FrameAdapter* frame_adapter_ = nullptr;
    WorldPointAdapter* world_point_adapter_ = nullptr;
    StaticInitializer static_initializer_;
};

}  // namespace DeltaVins
