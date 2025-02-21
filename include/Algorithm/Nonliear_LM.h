#pragma once
#include <limits>
#include <Eigen/Dense>

struct LM_Result {
    bool bConverged;
    float cost;
    float dZMax;
    float bMax;
    void clear() {
        bConverged = false;
        cost = std::numeric_limits<float>::max();
        dZMax = cost;
        bMax = cost;
    }
    friend std::ostream& operator<<(std::ostream& s, const LM_Result& p) {
        s << "Converged:" << p.bConverged << " cost:" << p.cost << " dzMax"
          << p.dZMax << " bMax" << p.bMax << std::endl;
        return s;
    }
};

template <int nDim, typename Type>
class NonLinear_LM {
   public:
    NonLinear_LM(Type Epsilon1, Type Epsilon2, Type tau, int nMaxIters,
                 bool verbose = false)
        : max_b_(Epsilon1),
          max_dz_(Epsilon2),
          max_iters_(nMaxIters),
          tau_(tau),
          verbose_(verbose) {

          };
    ~NonLinear_LM() {}
    void clear();
    virtual Type EvaluateF(bool bNewZ, Type huberThresh) = 0;
    virtual bool UserDefinedDecentFail() = 0;
    virtual bool UserDefinedConvergeCriteria() = 0;
    void solve();

    LM_Result m_Result;

   protected:
    Eigen::Matrix<Type, nDim, nDim> H, HTemp;
    Eigen::Matrix<Type, nDim, 1> b, bTemp;
    Eigen::Matrix<Type, nDim, 1> z, zNew;
    Eigen::Matrix<Type, nDim, 1> dZ;
    Type max_b_, max_dz_;
    int max_iters_;
    int num_iter_;
    Type tau_;
    bool verbose_;
};

template <int nDim, typename Type>
void NonLinear_LM<nDim, Type>::clear() {
    H.setZero();
    b.setZero();
    HTemp.setZero();
    bTemp.setZero();
    z.setZero();
    dZ.setZero();
    m_Result.clear();
}

template <int nDim, typename Type>
void NonLinear_LM<nDim, Type>::solve() {
    float nu = 2;

    float cost0 = EvaluateF(false, 10.f);
    H = HTemp;
    b = bTemp;
    float mu = H.diagonal().maxCoeff() * tau_;
    if (b.cwiseAbs().maxCoeff() < max_b_) {
        m_Result.bMax = b.cwiseAbs().maxCoeff();
        m_Result.bConverged = true;
    }

    for (num_iter_ = 1; !m_Result.bConverged && num_iter_ < max_iters_;
         ++num_iter_) {
        Eigen::Matrix<Type, nDim, nDim> H_ =
            H + Eigen::Matrix<Type, nDim, nDim>::Identity(3, 3) * mu;

        dZ = H_.ldlt().solve(b);

        if (dZ.norm() < max_dz_ * (z.norm() + max_dz_)) {
            m_Result.bConverged = true;
            if (verbose_)
                LOGI("\t%.6f / %.6f\n", dZ.norm(),
                     max_dz_ * (z.norm() + max_dz_));
        }

        zNew = z + dZ;

        if (verbose_) {
            LOGI("#Iter:\t %02d\n", num_iter_);
            LOGI("\t#mu:%.6f\n", mu);
            LOGI("\t#dZ:");
            for (int i = 0; i < nDim; ++i) {
                LOGI("  %.6f", dZ[i]);
            }
            LOGI("\n\t#z:\t");
            for (int i = 0; i < nDim; ++i) {
                LOGI("  %.6f", z[i]);
            }
            LOGI("  ->  ");
            for (int i = 0; i < nDim; ++i) {
                LOGI("  %.6f", zNew[i]);
            }
            LOGI("\n");
            LOGI("\n\t#b:\t");
            for (int i = 0; i < nDim; ++i) {
                LOGI("  %.6f", b[i]);
            }
            LOGI("\n");
        }

        // compute rho for estimating decent performance
        float cost1 = EvaluateF(true, num_iter_ > 2 ? 3.f : 10.f);
        float rho = (cost0 - cost1) / (0.5 * dZ.dot(mu * dZ + b));
        if (verbose_) LOGI("\trho:%.6f\n", rho);
        if (rho > 0 && !UserDefinedDecentFail()) {
            z = zNew;
            H = HTemp;
            b = bTemp;
            if (verbose_) {
                LOGI("\t#cost:\t %.6f  ->  %.6f\n", cost0, cost1);
                LOGI("\t#Status: Accept\n");
            }
            cost0 = cost1;

            m_Result.bMax = b.cwiseAbs().maxCoeff();
            m_Result.dZMax = dZ.cwiseAbs().maxCoeff();
            m_Result.cost = cost0;

            if (b.cwiseAbs().maxCoeff() < max_b_) m_Result.bConverged = true;
            mu = mu *
                 std::max(1.f / 3.f, 1.f - float(std::pow(2.f * rho - 1.f, 3)));
            nu = 2;

        } else {
            mu = mu * nu;
            nu = 2 * nu;
            if (verbose_) LOGI("\t #Status: Reject\n");
        }

        if (verbose_)
            LOGI(
                "--------------------------------------------------------------"
                "\n");
    }
    UserDefinedConvergeCriteria();
}