#pragma once
#include <Algorithm/VIO_Constexprs.h>

#include "dataStructure/filterStates.h"
#include "dataStructure/vioStructures.h"

namespace DeltaVins {

struct ImuPreintergration;
struct PointState;
struct CamState;

class SquareRootEKFSolver {
   public:
    friend class VIOAlgorithm;

    SquareRootEKFSolver();

    void Init(CamState *state, Vector3f *vel, bool *static_);
    void AddCamState(CamState *state);

    void Propagate(const ImuPreintergration *pImuTerm);

    void PropagateStatic(const ImuPreintergration *pImuTerm);

    void PropagateNew(const ImuPreintergration *pImuTerm);

    void Marginalize();

    void MarginalizeNew();

    void MarginalizeStatic();

    void MarginalizeGivens();

    bool MahalanobisTest(PointState *state);

    int ComputeJacobians(Landmark *track);

    void AddMsckfPoint(PointState *state);

    void AddSlamPoint(PointState *state);

    void AddVelocityConstraint(int nRows);

    int _AddPlaneContraint();
    int StackInformationFactorMatrix();

    void SolveAndUpdateStates();

    int _AddNewSlamPointConstraint();

    int AddSlamPointConstraint();

   private:
    void _UpdateByGivensRotations(int row, int col);

    int _AddPositionContraint(int nRows);

#ifdef PLATFORM_ARM
    void _updateByGivensRotationsNeon();

#endif
    void _MarginByGivensRotation();

    // MatrixMf m_infoFactorInverseMatrix;

    MatrixMfR info_factor_matrix_to_marginal_;

    MatrixOfR stacked_matrix_;
    VectorOf obs_residual_;

    int stacked_rows_ = 0;

    MatrixMf info_factor_matrix_;  // Upper Triangle Matrix

    MatrixMf info_factor_matrix_after_mariginal_;
    VectorMf residual_;

    int CURRENT_DIM = 0;
    std::vector<CamState *> cam_states_;
    std::vector<PointState *> msckf_points_;
    std::vector<PointState *> slam_point_;
    std::vector<PointState *> new_slam_point_;
    Vector3f *vel_;
    bool *static_;
    CamState *new_state_ = nullptr;
    CamState *last_state_ = nullptr;
};
}  // namespace DeltaVins
