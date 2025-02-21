#pragma once
#include <memory>
#include <unordered_set>
#include <vector>

#include "Algorithm/Nonliear_LM.h"
#include "filterStates.h"
#include "utils/typedefs.h"
#include <opencv2/opencv.hpp>

namespace DeltaVins {
struct Frame;

struct Landmark;

struct VisualObservation {
    using Ptr = std::shared_ptr<VisualObservation>;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VisualObservation(const Vector2f& px, Frame* frame);

    Vector2f px_reprj;            // reprojected position only used for debug
    Vector2f px;                  // feature position
    Vector3f ray_in_imu;          // camera ray
    Frame* link_frame = nullptr;  // pointer to linked frame
    Landmark* link_landmark = nullptr;  // pointer to linked landmark
};

struct Frame {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Frame(int sensor_id);

    ~Frame();

    void RemoveAllObservations();
    VisualObservation::Ptr AddVisualObservation(const Vector2f& px);

    cv::Mat image;  // image captured from camera

    int sensor_id;
    // int cam_id;    // left or right
    int frame_id;  // frame id

    CamState* state =
        nullptr;  // pointer to camera states including position,rotation,etc.

    std::unordered_set<VisualObservation::Ptr>
        visual_obs;  // All tracked feature in this frame.
    int valid_landmark_num;
    int64_t timestamp;

    bool flag_keyframe;
    using Ptr = std::shared_ptr<Frame>;
};

struct Landmark : public NonLinear_LM<3, double> {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct VisualObservationComparator {
        bool operator()(const VisualObservation::Ptr& a,
                        const VisualObservation::Ptr& b) const {
            return a->link_frame->frame_id < b->link_frame->frame_id;
        }
    };

    Landmark();

    ~Landmark();

    bool flag_dead;          // still be tracked
    int flag_dead_frame_id;  // used for debug
    int num_obs;             // the number of observations matched
    float ray_angle;   // ray angle between current camera ray and the first ray
    float ray_angle0;  // last ray angle
    int landmark_id_;  // used for debug
    Vector3f Pw_stereo_prior;
    bool has_stereo_prior;

    std::set<VisualObservation::Ptr, VisualObservationComparator>
        // std::unordered_set<VisualObservation::Ptr>
        visual_obs;                    // vector of all visual observations
    VisualObservation::Ptr last_obs_;  // pointer to the last observation
    VisualObservation::Ptr
        last_last_obs_;  // pointer to the second last observation
    Vector2f
        predicted_px;  // predicted pixel position using propagated camera pose
    PointState* point_state_;  // pointer to point state
    Frame* host_frame;
    bool flag_slam_point_candidate;
    std::vector<Matrix3d> dRs;  // used in Triangulate
    std::vector<Vector3d> dts;

    void AddVisualObservation(VisualObservation::Ptr obs);
    void AddVisualObservation(const Vector2f& px, Frame* frame, float depth);
    void RemoveVisualObservation(VisualObservation::Ptr obs);
    void PopObservation();

    void RemoveLinksInCamStates();
    void DrawFeatureTrack(cv::Mat& image, cv::Scalar color) const;
    float Reproject(bool verbose = true);
    void DrawObservationsAndReprojection(int time = 0);
    void PrintObservations();

    void RemoveUselessObservationForSlamPoint();

    // Triangulation
    bool UserDefinedConvergeCriteria() override;
    void PrintPositions();
    bool Triangulate();
    bool TriangulateLM(float depth_prior);
    bool TriangulationAnchorDepth(float& anchor_depth);
    bool StereoTriangulate();
    double EvaluateF(bool bNewZ, double huberThresh) override;
    bool UserDefinedDecentFail() override;
    using Ptr = std::shared_ptr<Landmark>;
};

}  // namespace DeltaVins
