#include "dataStructure/vioStructures.h"

#include <set>
#include <unordered_map>

#include "Algorithm/Initializer/Triangulation.h"
#include "Algorithm/vision/camModel/camModel.h"
#include "precompile.h"
#include "utils/SensorConfig.h"
#include "utils/utils.h"

namespace DeltaVins {

VisualObservation::VisualObservation(const Vector2f& px, Frame* frame)
    : px(px), link_frame(frame) {
    auto camModel = SensorConfig::Instance().GetCamModel(frame->sensor_id);
    ray_in_imu = camModel->imageToImu(px);
}

Frame::Frame(int sensor_id) {
    state = new CamState();
    state->host_frame = this;
    valid_landmark_num = 0;
    this->sensor_id = sensor_id;
    flag_keyframe = false;
    static std::unordered_map<int, int> frame_id_counter;
    frame_id = frame_id_counter[sensor_id]++;
}

VisualObservation::Ptr Frame::AddVisualObservation(const Vector2f& px) {
    auto obs = std::make_shared<VisualObservation>(px, this);
    visual_obs.insert(obs);
    return obs;
}

Frame::~Frame() {
    if (state) delete state;
}

void Frame::RemoveAllObservations() {
    for (auto obs : visual_obs) {
        Landmark* landmark = obs->link_landmark;
        if (landmark) {
            landmark->RemoveVisualObservation(obs);
        }
    }
    visual_obs.clear();
    valid_landmark_num = 0;
}

void Landmark::RemoveVisualObservation(VisualObservation::Ptr obs) {
    visual_obs.erase(obs);
    obs->link_landmark = nullptr;
    obs->link_frame->valid_landmark_num--;
}

Landmark::~Landmark() {
    if (!visual_obs.empty()) {
        RemoveLinksInCamStates();
    }
    if (point_state_) {
        delete point_state_;
        point_state_ = nullptr;
    }
}

Landmark::Landmark() : NonLinear_LM(1e-2, 0.005, 1e-3, 15, false) {
    flag_dead = false;
    num_obs = 0;
    ray_angle = 0;
    // last_moved_px = 0;
    point_state_ = nullptr;
    flag_slam_point_candidate = false;
    host_frame = nullptr;

    flag_dead_frame_id = -1;
    static int counter = 0;
    landmark_id_ = counter++;
}

bool Landmark::TriangulationAnchorDepth(float& anchor_depth) {
    std::vector<Eigen::Vector3d> ray_in_c;
    std::vector<Eigen::Matrix3d> Rwc;
    std::vector<Eigen::Vector3d> Pc_in_w;

    // Todo: support multi-camera
    CamModel::Ptr camModel = SensorConfig::Instance().GetCamModel(0);
    static Vector3d Tci = camModel->getTci().cast<double>();

    for (auto& visualOb : visual_obs) {
        ray_in_c.push_back(visualOb->ray_in_imu.cast<double>());
        Rwc.push_back(visualOb->link_frame->state->Rwi.cast<double>());
        Pc_in_w.push_back(visualOb->link_frame->state->Pwi.cast<double>() -
                          Tci);
    }
    Eigen::Vector3d Pw_triangulated;
    bool success = DeltaVins::TriangulationAnchorDepth(ray_in_c, Rwc, Pc_in_w,
                                                       Pw_triangulated);
    if (success) {
        Eigen::Vector3d Pw_triangulated_cam =
            Rwc[0].transpose() * (Pw_triangulated - Pc_in_w[0]);
        anchor_depth = Pw_triangulated_cam.z();
        //         if (point_state_ == nullptr) {
        //             point_state_ = new PointState();
        //             point_state_->host = this;
        //         }
        //         point_state_->Pw = Pw_triangulated.cast<float>();
        //         point_state_->Pw_FEJ = point_state_->Pw;
        //         m_Result.cost = Reproject(false);
        //         printf("TriangulationAnchorDepth cost: %f\n", m_Result.cost);
        //         if (m_Result.cost < 2 && !flag_dead) {
        //             flag_slam_point_candidate = true;
        //             printf("slam candidate with obs num: %d\n",
        //             visual_obs.size());
        //         }
        return true;
    }
    return false;
}
bool Landmark::Triangulate() {
    float anchor_depth = 0;
    if (!TriangulationAnchorDepth(anchor_depth)) return false;
    return TriangulateLM(anchor_depth);
}

bool Landmark::TriangulateLM(float depth_prior) {
    if (verbose_) LOGI("###PointID:%d", landmark_id_);
    // if (point_state_) return m_Result.bConverged;
    // TODO: multi-camera triangulation
    CamModel::Ptr camModel = SensorConfig::Instance().GetCamModel(0);
    static Vector3d Tci = camModel->getTci().cast<double>();

    clear();

    auto& leftVisualOb = *visual_obs.begin();
    // Vector3f pInImu = Rci.transpose() * (leftVisualOb.m_Ray_cam*2) + Pic;
    z = leftVisualOb->ray_in_imu.cast<double>();
    z /= z[2];
    z[2] = 1 / depth_prior;
    /*z.z() = 1/pInImu.z();*/

    const int nSize = visual_obs.size();
    dRs.resize(nSize);
    dts.resize(nSize);
    Matrix3d leftR = leftVisualOb->link_frame->state->Rwi.cast<double>();
    Vector3d leftP = leftVisualOb->link_frame->state->Pwi.cast<double>();
    {
        int i = 0;
        for (auto& visualOb : visual_obs) {
            Matrix3d R_i = visualOb->link_frame->state->Rwi.cast<double>();
            Vector3d P_i = visualOb->link_frame->state->Pwi.cast<double>();
            dRs[i] = R_i.transpose() * leftR;
            dts[i] = R_i.transpose() * (leftP - P_i) + Tci - dRs[i] * Tci;
            i++;
        }
    }

    solve();

    if (point_state_ == nullptr) {
        point_state_ = new PointState();
        point_state_->host = this;
    }
    Vector3d cpt = z / z[2];
    cpt[2] = 1.0 / z[2];
    point_state_->Pw = (leftR * (cpt - Tci) + leftP).cast<float>();
    point_state_->Pw_FEJ = point_state_->Pw;
    float depthRatio = 0.01;
    if (m_Result.bConverged && !flag_dead && m_Result.cost < 2 &&
        H(2, 2) > depthRatio * H(0, 0) && H(2, 2) > depthRatio * H(1, 1))
        flag_slam_point_candidate = true;

    // if (z[2] < 0.1) flag_slam_point_candidate = false;

    dRs.clear();
    dts.clear();
    m_Result.cost = Reproject(false);
    if (m_Result.cost > 5) {
        delete point_state_;
        point_state_ = nullptr;
        return false;
    }

    return m_Result.bConverged;
}

double Landmark::EvaluateF(bool bNewZ, double huberThresh) {
    double cost = 0.0;
    // const int nSize = visual_obs.size();
    Matrix23d J23;
    Matrix3d J33;
    Vector3d position;

    // float huberCutTh = 10.f;

    if (bNewZ) {
        position = zNew / zNew[2];
        position[2] = 1.0 / zNew[2];
    } else {
        position = z / z[2];
        position[2] = 1.0 / z[2];
    }
    HTemp.setZero();
    bTemp.setZero();
    J33 << position[2], 0, -position[0] * position[2], 0, position[2],
        -position[1] * position[2], 0, 0, -position[2] * position[2];
    CamModel::Ptr camModel = SensorConfig::Instance().GetCamModel(0);
    int i = 0;
    for (auto& visualOb : visual_obs) {
        Vector3f p_cam_f = (dRs[i] * position + dts[i]).cast<float>();

        Matrix23f J23f;
        visualOb->px_reprj = camModel->camToImage(p_cam_f, J23f);
        Matrix23d J23d = J23f.cast<double>();

        Vector2d r = (visualOb->px - visualOb->px_reprj).cast<double>();
#if HUBER || 1
        double reprojErr = r.norm();
        double hw = reprojErr > huberThresh ? huberThresh / reprojErr : 1;
        // cost += reprojErr>huberThresh ?
        // huberThresh*(2*reprojErr-huberThresh):reprojErr*reprojErr;
        if (reprojErr > huberThresh) {
            cost += huberThresh * (2 * reprojErr - huberThresh);
        } else
            cost += reprojErr * reprojErr;
#else
        float hw = 1.0f;
        cost += r.squaredNorm();
#endif

        Matrix23d J = J23d * dRs[i] * J33;

        HTemp.noalias() += J.transpose() * J * hw;
        bTemp.noalias() += J.transpose() * r * hw;

        i++;
    }
    // if(!bNewZ)
    //	H += Matrix3f::Identity()*FLT_EPSILON;
    return cost;
}

bool Landmark::UserDefinedDecentFail() { return zNew[2] < 0; }

void Landmark::AddVisualObservation(VisualObservation::Ptr obs) {
    flag_dead = false;

    // TODO: moved to tracker
    // if (!visual_obs.empty()) {
    //     last_moved_px = (visual_obs.back().px - px).squaredNorm();
    // }
    num_obs++;

    // CamModel::Ptr camModel = SensorConfig::Instance().GetCamModel(0);
    // Vector3f ray0 = camModel->imageToImu(px);

    obs->link_landmark = this;
    obs->link_frame->valid_landmark_num++;
    visual_obs.insert(obs);
    last_last_obs_ = last_obs_;
    last_obs_ = obs;

    // Here we compute the ray angle between the former rays and the last ray
    // The ray angle is used to judge if the landmark is a good landmark
    // If the ray angle is too small, it means the landmark is far from the
    // camera and it is not a good landmark maybe it is not a good idea to use
    // the ray angle to judge the landmark?? because the ray angle is not a good
    // metric to describe the landmark
    ray_angle0 = ray_angle;
    Vector3f ray1 = obs->link_frame->state->Rwi * obs->ray_in_imu;

    float minDot = 2;
    for (auto& visualOb : visual_obs) {
        float dot = ((visualOb->link_frame->state->Rwi * visualOb->ray_in_imu)
                         .normalized())
                        .dot(ray1.normalized());
        if (dot < minDot) minDot = dot;
    }
    if (minDot > 0 && minDot < 1) {
        ray_angle = std::max(ray_angle, acosf(minDot));
    }
}

void Landmark::DrawFeatureTrack(cv::Mat& image, cv::Scalar color) const {
    std::set<VisualObservation::Ptr, VisualObservationComparator>
        visual_obs_set(visual_obs.begin(), visual_obs.end());

    VisualObservation::Ptr front_obs = nullptr;
    VisualObservation::Ptr curr_obs = nullptr;
    for (auto& visualOb : visual_obs_set) {
        curr_obs = visualOb;
        if (front_obs == nullptr) {
            front_obs = curr_obs;
            continue;
        }
        if ((curr_obs->px - front_obs->px).squaredNorm() > 900) {
            front_obs = curr_obs;
            continue;
        }
        cv::line(image, cv::Point(front_obs->px.x(), front_obs->px.y()),
                 cv::Point(curr_obs->px.x(), curr_obs->px.y()), _GREEN_SCALAR,
                 1);
        cv::circle(image, cv::Point(curr_obs->px.x(), curr_obs->px.y()), 2,
                   color);
        front_obs = curr_obs;
    }
    cv::circle(image, cv::Point(last_obs_->px.x(), last_obs_->px.y()), 8,
               color);
}
float Landmark::Reproject(bool verbose) {
    assert(point_state_);
    CamModel::Ptr camModel = SensorConfig::Instance().GetCamModel(0);
    float reprojErr = 0;
    for (auto& ob : visual_obs) {
        ob->px_reprj = camModel->imuToImage(
            ob->link_frame->state->Rwi.transpose() *
            (point_state_->Pw - ob->link_frame->state->Pwi));
        reprojErr += (ob->px_reprj - ob->px).norm();
    }
    reprojErr = visual_obs.size() > 0 ? reprojErr / visual_obs.size() : 100;
    if (reprojErr > 3) {
        if (reprojErr > 5)
            // DrawObservationsAndReprojection();
            if (verbose)
                printf("Reproj Err:%f, Triangulation Cost:%f\n", reprojErr,
                       m_Result.cost);
    }
    return reprojErr;
}

void Landmark::DrawObservationsAndReprojection(int time) {
#if ENABLE_VISUALIZER && !defined(PLATFORM_ARM)
    if (Config::NoGUI) return;
    cv::Mat display;
    bool first = 1;
    for (auto& ob : visual_obs) {
        cv::cvtColor(ob->link_frame->image, display, cv::COLOR_GRAY2BGR);
        if (first) {
            cv::circle(display, cv::Point(ob->px.x(), ob->px.y()), 8,
                       _GREEN_SCALAR);
            cv::circle(display, cv::Point(ob->px_reprj.x(), ob->px_reprj.y()),
                       10, _BLUE_SCALAR);
            first = 0;
        } else {
            cv::circle(display, cv::Point(ob->px.x(), ob->px.y()), 4,
                       _GREEN_SCALAR);
            cv::circle(display, cv::Point(ob->px_reprj.x(), ob->px_reprj.y()),
                       6, _BLUE_SCALAR);
        }
        cv::imshow("ob and reproj", display);
        cv::waitKey(time);
    }
#else
    (void)time;
#endif
}

void Landmark::PrintObservations() {
    for (auto& visualOb : visual_obs) {
        LOGI("Px: %f %f,Pos:%f %f %f", visualOb->px.x(), visualOb->px.y(),
             visualOb->link_frame->state->Pwi.x(),
             visualOb->link_frame->state->Pwi.y(),
             visualOb->link_frame->state->Pwi.z());
    }
}
void Landmark::RemoveUselessObservationForSlamPoint() {
    // return;
    // I think this function is not needed
    // assert(point_state_ && point_state_->flag_slam_point);
    for (auto iter = visual_obs.begin(); iter != visual_obs.end();) {
        auto visualOb = *iter;
        if (!visualOb->link_frame->flag_keyframe) {
            visualOb->link_landmark = nullptr;
            visualOb->link_frame->valid_landmark_num--;
            iter = visual_obs.erase(iter);
        } else {
            iter++;
        }
    }
}
bool Landmark::UserDefinedConvergeCriteria() {
    if (m_Result.cost < 10) m_Result.bConverged = true;
    if (m_Result.cost > 40) m_Result.bConverged = false;
    return true;
}

void Landmark::PrintPositions() {}

void Landmark::PopObservation() {
    flag_dead = true;
    visual_obs.erase(last_obs_);
    last_obs_->link_landmark = nullptr;
    last_obs_ = last_last_obs_;
    last_last_obs_ = nullptr;
    // visual_obs.back()->link_landmark = nullptr;
    ray_angle = ray_angle0;
    // visual_obs.pop_back();
    num_obs--;
}

void Landmark::RemoveLinksInCamStates() {
    for (auto& ob : visual_obs) {
        ob->link_landmark = nullptr;
        ob->link_frame->valid_landmark_num--;
    }
    visual_obs.clear();
}

}  // namespace DeltaVins
