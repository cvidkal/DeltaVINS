#include "Algorithm/DataAssociation/DataAssociation.h"

#include <random>

#include "Algorithm/DataAssociation/TwoPointRansac.h"
#include "Algorithm/solver/SquareRootEKFSolver.h"
#include "Algorithm/vision/camModel/camModel.h"
#include "dataStructure/Grid.h"
#include "dataStructure/vioStructures.h"
#include "precompile.h"
#include "utils/SensorConfig.h"
#include "utils/utils.h"
#define MAX_MSCKF_FEATURE_UPDATE_PER_FRAME MAX_POINT_SIZE

namespace DeltaVins {
namespace DataAssociation {

TwoPointRansac* g_two_point_ransac = nullptr;
SquareRootEKFSolver* g_square_root_solver = nullptr;
std::vector<LandmarkPtr> g_tracked_feature_to_update;
std::vector<LandmarkPtr> g_tracked_feature_next_update;
std::vector<std::vector<LandmarkPtr>> g_grid22;

cv::Mat reprojImage;
cv::Mat reprojImage2;

void Clear() {
    g_tracked_feature_next_update.clear();
    g_tracked_feature_to_update.clear();
    delete g_two_point_ransac;
}

void DrawPointsAfterUpdates(std::vector<PointState*>& m_PointStates,
                            int cam_id) {
    if (Config::NoGUI) return;
    reprojImage2 = cv::Mat::zeros(480, 640, CV_8UC3);
    for (auto& p : g_tracked_feature_to_update) {
        p->Reproject();
        for (auto& ob : p->visual_obs[cam_id]) {
            cv::circle(reprojImage2, cv::Point(ob->px.x(), ob->px.y()), 4,
                       _BLUE_SCALAR);
            cv::circle(reprojImage2,
                       cv::Point(ob->px_reprj.x(), ob->px_reprj.y()), 4,
                       _RED_SCALAR);
            cv::line(reprojImage2, cv::Point(ob->px.x(), ob->px.y()),
                     cv::Point(ob->px_reprj.x(), ob->px_reprj.y()),
                     _GREEN_SCALAR);
        }
        // for (size_t i = 0; i < p->visual_obs.size() - 1; ++i) {
        //     cv::line(
        //         reprojImage2,
        //         cv::Point(p->visual_obs[i]->px.x(),
        //         p->visual_obs[i]->px.y()), cv::Point(p->visual_obs[i +
        //         1]->px.x(),
        //                   p->visual_obs[i + 1]->px.y()),
        //         _GREEN_SCALAR);
        // }
    }
    for (auto& p2 : m_PointStates) {
        auto p = p2->host;
        p->Reproject();
        auto& ob = p->last_obs_[cam_id];
        cv::circle(reprojImage2, cv::Point(ob->px.x(), ob->px.y()), 10,
                   _BLUE_SCALAR);
        cv::circle(reprojImage2, cv::Point(ob->px_reprj.x(), ob->px_reprj.y()),
                   10, _RED_SCALAR);
        cv::line(reprojImage2, cv::Point(ob->px.x(), ob->px.y()),
                 cv::Point(ob->px_reprj.x(), ob->px_reprj.y()), _GREEN_SCALAR);
    }
    // cv::imshow("Points After Updates", reprojImage2);
}
void DrawPointsBeforeUpdates(std::vector<PointState*>& m_PointStates,
                             int cam_id) {
    if (Config::NoGUI) return;
    reprojImage = cv::Mat::zeros(480, 640, CV_8UC3);

    for (auto& p : g_tracked_feature_to_update) {
        p->Reproject();
        for (auto& ob : p->visual_obs[cam_id]) {
            cv::circle(reprojImage, cv::Point(ob->px.x(), ob->px.y()), 4,
                       _BLUE_SCALAR);
            cv::circle(reprojImage,
                       cv::Point(ob->px_reprj.x(), ob->px_reprj.y()), 4,
                       _RED_SCALAR);
            cv::line(reprojImage, cv::Point(ob->px.x(), ob->px.y()),
                     cv::Point(ob->px_reprj.x(), ob->px_reprj.y()),
                     _GREEN_SCALAR);
        }
        // for (size_t i = 0; i < p->visual_obs.size() - 1; ++i) {
        //     cv::line(
        //         reprojImage,
        //         cv::Point(p->visual_obs[i]->px.x(),
        //         p->visual_obs[i]->px.y()), cv::Point(p->visual_obs[i +
        //         1]->px.x(),
        //                   p->visual_obs[i + 1]->px.y()),
        //         _GREEN_SCALAR);
        // }
    }

    for (auto& p2 : m_PointStates) {
        auto p = p2->host;
        p->Reproject();
        auto& ob = p->last_obs_[cam_id];
        cv::circle(reprojImage, cv::Point(ob->px.x(), ob->px.y()), 10,
                   _BLUE_SCALAR);
        cv::circle(reprojImage, cv::Point(ob->px_reprj.x(), ob->px_reprj.y()),
                   10, _RED_SCALAR);
        cv::line(reprojImage, cv::Point(ob->px.x(), ob->px.y()),
                 cv::Point(ob->px_reprj.x(), ob->px_reprj.y()), _GREEN_SCALAR);
    }
    // cv::imshow("Points Before Updates", reprojImage);
}

void InitDataAssociation(SquareRootEKFSolver* solver) {
    if (!g_two_point_ransac) g_two_point_ransac = new TwoPointRansac();
    g_square_root_solver = solver;
    g_grid22.resize(4);
}

int RemoveOutlierBy2PointRansac(Matrix3f& dR,
                                std::list<LandmarkPtr>& vTrackedFeatures,
                                int sensor_id, int cam_id) {
    assert(g_two_point_ransac);

    std::vector<Vector3f> ray0, ray1;
    std::vector<Vector2f> p0, p1;
    std::vector<Landmark*> goodTracks;
    ray0.reserve(600);
    ray1.reserve(600);
    p0.reserve(600);
    p1.reserve(600);
    goodTracks.reserve(600);
    int nGoodPoints = 0;

    for (const auto& tracked_feature : vTrackedFeatures) {
        if (tracked_feature->flag_dead[cam_id]) continue;
        // int nObs = tracked_feature->visual_obs.size();
        auto& lastOb = tracked_feature->last_obs_[cam_id];
        ray1.push_back(lastOb->ray_in_cam);
        p1.push_back(lastOb->px);
        auto& lastSecondOb = tracked_feature->last_last_obs_[cam_id];
        ray0.push_back(lastSecondOb->ray_in_cam);
        p0.push_back(lastSecondOb->px);
        goodTracks.push_back(tracked_feature.get());
    }

    std::vector<bool> vInliers;
    g_two_point_ransac->FindInliers(p0, ray0, p1, ray1, dR, vInliers,
                                    sensor_id);
    for (int i = 0, n = vInliers.size(); i < n; ++i) {
        if (!vInliers[i]) {
            auto& track = goodTracks[i];

            track->flag_dead_frame_id[cam_id] =
                track->last_obs_[cam_id]->link_frame->state->m_id;
            track->PopObservation(cam_id);
            continue;
        }
        nGoodPoints++;
    }
    return nGoodPoints;
}

void _addBufferPoints(std::vector<std::shared_ptr<Landmark>>& vDeadFeature) {
    constexpr int MAX_BUFFER_OBS = 5;
    for (auto trackedFeature : g_tracked_feature_next_update) {
        if (trackedFeature->point_state_)
            assert(!trackedFeature->point_state_->flag_slam_point);
        if (trackedFeature->valid_obs_num > MAX_BUFFER_OBS)
            vDeadFeature.push_back(trackedFeature);
        else
            trackedFeature->RemoveLinksInCamStates();
    }
#if OUTPUT_DEBUG_INFO
    printf("  Add Buffer Points:%d/%d\n", vDeadFeature.size(),
           g_tracked_feature_next_update.size());
#endif
    g_tracked_feature_next_update.clear();
}

void _addDeadPoints(std::list<LandmarkPtr>& vTrackedFeatures,
                    std::vector<std::shared_ptr<Landmark>>& vDeadFeature) {
    constexpr int MIN_OBS = 4;
    int nDeadPoints2Updates = 0;
    int nDeadPointsAbandoned = 0;
    int nAlivePoints2Updates = 0;
    constexpr int MIN_OBS_ALIVE = 6;
    constexpr int MIN_OBS_TRACKED = 6;
    for (auto iter = vTrackedFeatures.begin();
         iter != vTrackedFeatures.end();) {
        auto tracked_feature = *iter;

        if (tracked_feature->point_state_ &&
            tracked_feature->point_state_->flag_slam_point) {
            ++iter;
            continue;
        }

        if (tracked_feature->flag_dead_all) {
            if (tracked_feature->num_obs_tracked >= MIN_OBS_TRACKED &&
                tracked_feature->valid_obs_num >= MIN_OBS) {
                vDeadFeature.push_back(tracked_feature);
                nDeadPoints2Updates++;
            } else {
                tracked_feature->RemoveLinksInCamStates();
                nDeadPointsAbandoned++;
            }
            iter = vTrackedFeatures.erase(iter);
            continue;
        }
        if (tracked_feature->valid_obs_num >= MIN_OBS_ALIVE &&
            tracked_feature->num_obs_tracked > MIN_OBS_TRACKED) {
            nAlivePoints2Updates++;
            vDeadFeature.push_back(tracked_feature);
        }
        ++iter;
    }

#if OUTPUT_DEBUG_INFO
    printf("  Dead Points: %d Good / %d Bad\n", nDeadPoints2Updates,
           nDeadPointsAbandoned);
    printf("  AlivePoints to Update:%d\n", nAlivePoints2Updates);
#endif
}

void _pushPoints2Grid(
    const std::vector<std::shared_ptr<Landmark>>& vDeadFeature) {
    static std::vector<std::vector<LandmarkPtr>> vvGrid44(4 * 4);
    static CamModel::Ptr camModel = SensorConfig::Instance().GetCamModel(0);
    static const int STEPX = camModel->width() / 4;
    static const int STEPY = camModel->height() / 4;

    auto comparator_less = [](const LandmarkPtr& a, const LandmarkPtr& b) {
        return a->flag_dead_all == b->flag_dead_all
                   ? a->ray_angle < b->ray_angle
                   : a->flag_dead_all < b->flag_dead_all;
    };

    auto selectTop2 = [&](const std::vector<LandmarkPtr>& src,
                          std::vector<LandmarkPtr>& dst) {
        LandmarkPtr pFirst = nullptr, pSecond = nullptr;
        if (!src.empty()) {
            for (auto tracked_feature : src) {
                if (!pSecond || comparator_less(pSecond, tracked_feature)) {
                    if (pSecond && pSecond->flag_dead_all)
                        g_tracked_feature_next_update.push_back(pSecond);
                    pSecond = tracked_feature;
                    if (!pFirst || comparator_less(pFirst, pSecond)) {
                        std::swap(pFirst, pSecond);
                    }
                } else {
                    if (tracked_feature->flag_dead_all)
                        g_tracked_feature_next_update.push_back(
                            tracked_feature);
                }
            }
            if (pSecond) dst.push_back(pSecond);
            if (pFirst && pFirst != pSecond) dst.push_back(pFirst);
        }
    };

    for (auto deadFeature : vDeadFeature) {
        // Todo: here right camera observation is handled with left camera
        // observation,
        //  need to properly handle the right camera observation
        auto& ob = deadFeature->last_obs_[0];
        if (ob) {
            vvGrid44[int(ob->px.x() / STEPX) + 4 * int(ob->px.y() / STEPY)]
                .push_back(deadFeature);
        } else {
            auto& ob = deadFeature->last_obs_[1];
            if (ob) {
                vvGrid44[int(ob->px.x() / STEPX) + 4 * int(ob->px.y() / STEPY)]
                    .push_back(deadFeature);
            }
        }
    }
#if OUTPUT_DEBUG_INFO
    printf("# 4*4 Dead Points:\n");
    printf("____________________\n");
    printf("|%03d|%03d|%03d|%03d|\n", vvGrid44[0].size(), vvGrid44[1].size(),
           vvGrid44[2].size(), vvGrid44[3].size());
    printf("|%03d|%03d|%03d|%03d|\n", vvGrid44[4].size(), vvGrid44[5].size(),
           vvGrid44[6].size(), vvGrid44[7].size());
    printf("|%03d|%03d|%03d|%03d|\n", vvGrid44[8].size(), vvGrid44[9].size(),
           vvGrid44[10].size(), vvGrid44[11].size());
    printf("|%03d|%03d|%03d|%03d|\n", vvGrid44[12].size(), vvGrid44[13].size(),
           vvGrid44[14].size(), vvGrid44[15].size());
    printf("____________________\n");
#endif

    static int LUT[16] = {0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3};

    for (int i = 0; i < 16; ++i) {
        selectTop2(vvGrid44[i], g_grid22[LUT[i]]);
    }

    for (auto& grid : g_grid22) {
        std::sort(grid.begin(), grid.end(), comparator_less);
    }

    for (auto& grid : vvGrid44) grid.clear();

#if OUTPUT_DEBUG_INFO
    printf("# 2*2 Dead Points:\n");
    printf("____________________\n");
    printf("|  %03d  |  %03d  |\n", g_grid22[0].size(), g_grid22[1].size());
    printf("|  %03d  |  %03d  |\n", g_grid22[2].size(), g_grid22[3].size());
    printf("____________________\n");
#endif
}

bool _tryAddMsckfPoseConstraint(const std::list<LandmarkPtr>& lTrackFeatures) {
    int nPointsPerGrid = MAX_MSCKF_FEATURE_UPDATE_PER_FRAME / 4;
    int nPointsLeft = MAX_MSCKF_FEATURE_UPDATE_PER_FRAME;
    int nPointsAllAdded = 0;
    int nPointsTriangleFailed = 0;
    int nPointsMahalaFailed = 0;
    int valid_points = 0;

    // Todo: Multi camera support
    int halfX = SensorConfig::Instance().GetCamModel(0)->width() / 2;
    int halfY = SensorConfig::Instance().GetCamModel(0)->height() / 2;
    // int nPointsSlamPerGrid = MAX_POINT_SIZE / 4;
    std::vector<int> vPointsSLAMLeft{0, 0, 0, 0};
    std::vector<int> vPointsSLAMNow{0, 0, 0, 0};
    static std::vector<std::vector<Landmark*>> m_slamPointGrid22(4);
    // static Grid<Landmark*, 2, 2> m_slamPointGrid22(halfX, halfY);
    int nSlamPoint = 0;

    // Step 1: Prepare the grid for slam points
    std::for_each(m_slamPointGrid22.begin(), m_slamPointGrid22.end(),
                  [](auto& a) { a.clear(); });
    for (auto& point : lTrackFeatures) {
        if (point->point_state_ && point->point_state_->flag_slam_point &&
            !point->flag_dead_all) {
            float x, y;
            if (point->last_obs_[0]) {
                x = point->last_obs_[0]->px.x();
                y = point->last_obs_[0]->px.y();
            } else {
                x = point->last_obs_[1]->px.x();
                y = point->last_obs_[1]->px.y();
            }
            if (x < halfX)
                if (y < halfY) {
                    vPointsSLAMNow[0]++;
                    m_slamPointGrid22[0].push_back(point.get());
                } else {
                    vPointsSLAMNow[2]++;
                    m_slamPointGrid22[2].push_back(point.get());
                }
            else if (y < halfY) {
                vPointsSLAMNow[1]++;
                m_slamPointGrid22[1].push_back(point.get());
            } else {
                m_slamPointGrid22[3].push_back(point.get());
                vPointsSLAMNow[3]++;
            }
            nSlamPoint++;
        }
    }
    int max_slam_point = 16;
    int nSlamPointsPerGrid = max_slam_point / 4;
    if (nSlamPoint < max_slam_point) {
        for (int i = nSlamPoint; i < max_slam_point; i++) {
            int k = 0;
            for (int j = 0; j < 3; j++) {
                k = vPointsSLAMNow[k] <= vPointsSLAMNow[j + 1] ? k : j + 1;
            }
            vPointsSLAMLeft[k]++;
            vPointsSLAMNow[k]++;
        }
    } else {
        for (int i = 0; i < 4; ++i) {
            if (vPointsSLAMNow[i] > nSlamPointsPerGrid) {
                m_slamPointGrid22[i][2]
                    ->point_state_->flag_to_next_marginalize = true;
                vPointsSLAMLeft[i]++;
            }
        }
    }

    valid_points += nSlamPoint;

    LOGD("SlamCnt:%d %d %d %d %d", nSlamPoint, vPointsSLAMLeft[0],
         vPointsSLAMLeft[1], vPointsSLAMLeft[2], vPointsSLAMLeft[3]);
    nPointsLeft =
        (MAX_OBS_SIZE - MAX_ADDITIONAL_MSCKF_POINT * MAX_WINDOW_SIZE * 2 -
         nSlamPoint * 5) /
        (MAX_WINDOW_SIZE * 2);
    // nPointsLeft = MAX_ALL_POINT_SIZE - nSlamPoint;
    nPointsPerGrid = nPointsLeft / 4;

    std::vector<int> vPointsLeft = {nPointsPerGrid, nPointsPerGrid,
                                    nPointsPerGrid, nPointsPerGrid};

    auto triangleAndVerify = [&](LandmarkPtr& track) {
        if (track->Triangulate()) {
#if OUTPUT_DEBUG_INFO
            printf("#### Triangulation Success\n");
#endif
            if (g_square_root_solver->ComputeJacobians(track.get())) {
                if (g_square_root_solver->MahalanobisTest(
                        track->point_state_)) {
                    nPointsAllAdded++;
                    return true;
                }
                nPointsMahalaFailed++;
                return false;
            }
            return false;
        }
        nPointsTriangleFailed++;
#if OUTPUT_DEBUG_INFO
        printf("#### Triangulation Fail\n");
#endif
        return false;
    };

    auto selectPoints = [&]() {
#if OUTPUT_DEBUG_INFO
        int ii = 0;
        int jInGrid = 0;

#endif
        for (int i = 0; i < 4; ++i) {
            auto& grid = g_grid22[i];
#if OUTPUT_DEBUG_INFO

            printf("## Grid %d in 2*2:\n", ii);
            ii++;
#endif
            while (vPointsLeft[i] && !grid.empty()) {
#if OUTPUT_DEBUG_INFO
                printf("### %d point in Grid %d\n", jInGrid++, ii);
#endif
                auto& ft = grid.back();
                if (triangleAndVerify(ft)) {
                    valid_points++;
                    --nPointsLeft;
#if OUTPUT_DEBUG_INFO
                    ++nPointsAllAdded;
#endif
                    if (!ft->flag_dead_all && ft->flag_slam_point_candidate &&
                        vPointsSLAMLeft[i]) {  // If it is a slam
                                               // point

                        g_square_root_solver->AddSlamPoint(ft->point_state_);
                        vPointsSLAMLeft[i]--;
                    } else {
                        g_square_root_solver->AddMsckfPoint(ft->point_state_);
                        ft->SetDeadFlag(true, -1);
                    }
                    vPointsLeft[i]--;

                    g_tracked_feature_to_update.push_back(ft);
                } else {  // if triangulation failed
                    //?? why here we need to set the flag_dead to false?
                    // should keep unchanged?
                    // ft->flag_dead = false;
                }
                grid.pop_back();
            }
        }
    };
    auto bufferPoints = [&]() {
        for (auto& grid : g_grid22) {
            for (auto& ft : grid) {
                if (ft->flag_dead_all) {
                    g_tracked_feature_next_update.push_back(ft);
                }
            }
            grid.clear();
        }
    };

    selectPoints();

    if (nPointsLeft) {
        int nMoreGrid = 0;
        for (auto& ptLeft : vPointsLeft) {
            if (ptLeft == 0) nMoreGrid++;
        }
        for (auto& ptLeft : vPointsLeft) {
            if (ptLeft == 0)
                ptLeft = nPointsLeft / nMoreGrid;
            else
                ptLeft = 0;
        }
        selectPoints();
    }
#if OUTPUT_DEBUG_INFO
    printf("### %d Points to Update\n", nPointsAllAdded);
    printf(
        "### nPointsAdded:%d nPointsTriangleFailed:%d "
        "nPointsMahalaFailed:%d\n",
        nPointsAllAdded, nPointsTriangleFailed, nPointsMahalaFailed);
#endif
    bufferPoints();
    return valid_points;
}

int _tryAddStereoPoint(const std::list<LandmarkPtr>& lTrackFeatures) {
    bool use_stereo = SensorConfig::Instance().GetCamModel(0)->IsStereo();
    if (!use_stereo) return 0;

    int points_added_total = 0;

    int x_grid_size = SensorConfig::Instance().GetCamModel(0)->width() / 4;
    int y_grid_size = SensorConfig::Instance().GetCamModel(0)->height() / 4;
    Grid<Landmark*, 4, 4> slam_point_grid44(x_grid_size, y_grid_size);
    int min_obs_tracked = 4;
    int num_points_candidate = 0;
    for (auto& point : lTrackFeatures) {
        if (point->flag_dead[0] || point->flag_dead[1]) continue;
        if (point->point_state_) continue;
        if (point->num_obs_tracked < min_obs_tracked) continue;
        float x = point->last_obs_[0]->px.x();
        float y = point->last_obs_[0]->px.y();
        slam_point_grid44.Add(point.get(), x, y);
        num_points_candidate++;
    }

    if (!num_points_candidate) {  // no candidate points
        return 0;
    }

    // sort the grid by the number of points
    std::vector<std::pair<int, int>> grid_points_nums;
    for (int i = 0; i < 16; ++i) {
        grid_points_nums.push_back({i, slam_point_grid44.Get(i).size()});
    }
    std::stable_sort(
        grid_points_nums.begin(), grid_points_nums.end(),
        [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
            return a.second < b.second;
        });

    int all_points_left = MAX_POINT_SIZE;
    int grid_num_left = 16;
    for (auto& grid_points_num : grid_points_nums) {
        auto& grid = slam_point_grid44.Get(grid_points_num.first);
        std::stable_sort(grid.begin(), grid.end(),
                         [](const Landmark* a, const Landmark* b) {
                             return a->stereo_parallax > b->stereo_parallax;
                         });
        int points_added = 0;
        int points_per_grid = all_points_left / grid_num_left;
        for (auto& point : grid) {
            if (points_added >= points_per_grid) {
                break;
            }
            if (point->Triangulate()) {
                if (g_square_root_solver->ComputeJacobians(point)) {
                    if (g_square_root_solver->MahalanobisTest(
                            point->point_state_)) {
                        points_added++;
                        if (point->flag_slam_point_candidate) {
                            g_square_root_solver->AddSlamPoint(
                                point->point_state_);
                        } else {
                            g_square_root_solver->AddMsckfPoint(
                                point->point_state_);
                            point->SetDeadFlag(true, -1);
                        }
                        points_added_total++;
                        all_points_left--;
                    }
                }
            }
        }
        grid_num_left--;
    }
    return points_added_total;
}

void DoDataAssociation(std::list<LandmarkPtr>& vTrackedFeatures, bool static_) {
    g_tracked_feature_to_update.clear();

#if USE_STATIC_DETECTION
    if (static_) {
        g_tracked_feature_next_update.clear();
        return;
    }

#endif
    static std::vector<LandmarkPtr> vDeadFeature;

    _addBufferPoints(vDeadFeature);

    _addDeadPoints(vTrackedFeatures, vDeadFeature);

    _pushPoints2Grid(vDeadFeature);

    vDeadFeature.clear();

    int valid_points = _tryAddMsckfPoseConstraint(vTrackedFeatures);
    if (!valid_points) {
        _tryAddStereoPoint(vTrackedFeatures);
    }
}

}  // namespace DataAssociation
}  // namespace DeltaVins
