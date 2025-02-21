#pragma once
#include <memory>

#include "utils/typedefs.h"

namespace DeltaVins {
struct Frame;
typedef std::shared_ptr<Frame> FramePtr;
struct Landmark;
typedef std::shared_ptr<Landmark> LandmarkPtr;

struct PointState {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Vector3f Pw;      //	point position in world frame
    Vector3f Pw_FEJ;  //	point position First Estimate Jacobian

    MatrixXfR H;  //	Observation Matrix
    Landmark* host = nullptr;

    bool flag_to_marginalize = false;
    bool flag_to_next_marginalize = false;
    bool flag_slam_point;
    int index_in_window;  // point idx in sliding window

    int m_id = 0;  // only used in visualizer
    int m_idVis = -1;
    PointState() {
        static int counter = 0;
        m_id = counter++;
        flag_slam_point = false;
    }
};

struct CamState {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Matrix3f Rwi;     // rotation matrix from imu frame to world frame
    Vector3f Pwi;     // imu position in world frame
    Vector3f Pw_FEJ;  // First Estimate Jacobian Imu Position in world frame

    int index_in_window;               // camera idx in sliding window
    bool flag_to_marginalize = false;  // flag to marginalize
    Frame* host_frame = nullptr;       // pointer to host frame
    Vector3f vel;

    int m_id = 0;  // only used in visualizer

    CamState() {
        static int counter = 0;
        m_id = counter++;
    }
};

struct MsckfState {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    std::vector<FramePtr> frames;  // All frames in sliding window
    Vector3f vel;                  // linear velocity
};

}  // namespace DeltaVins
