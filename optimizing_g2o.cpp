#include <iostream>

#include "g2o/stuff/sampler.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/batch_stats.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_dogleg.h"

#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/solvers/pcg/linear_solver_pcg.h"
#include "g2o/types/sba/types_six_dof_expmap.h"

#include "g2o/solvers/structure_only/structure_only_solver.h"
#include <ceres/ceres.h>
#include <ceres/internal/autodiff.h>
#include "common/projection.h"
#include "common/BALProblem.h"
#include "common/BundleParams_G2O.h"

/**
 * 相机位姿节点
 */
class VertexCameraBAL : public g2o::BaseVertex<9, Eigen::VectorXd> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VertexCameraBAL() {}

    bool read(std::istream &is) override {
        return false;
    }

    bool write(std::ostream &os) const override {
        return false;
    }

protected:
    void oplusImpl(const double *update) override {
        Eigen::VectorXd::ConstMapType update_vec(update, VertexCameraBAL::Dimension);
        this->_estimate += update_vec;
    }

    void setToOriginImpl() override {}
};

/**
 * 路标点节点
 * @return
 */

class VertexPointBAL : public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
    VertexPointBAL() {}

    bool read(std::istream &is) override {
        return false;
    }

    bool write(std::ostream &os) const override {
        return false;
    }

protected:
    void oplusImpl(const double *update) override {
        Eigen::Vector3d update_vec(update);
        this->_estimate += update_vec;
    }

    void setToOriginImpl() override {}

};

class EdgeObservationBAL : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexCameraBAL, VertexPointBAL> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeObservationBAL() {}

    template<typename T>
    bool operator()(const T *camera, const T *point, T *residuals) const {
        T prediction[2];
        // 将空间点根据相机位姿和内参投影到像平面
        CamProjectionWithDistortion(camera, point, prediction);
        // 计算观测值和预测值的差
        residuals[0] = prediction[0] - T(this->measurement()(0));
        residuals[1] = prediction[1] - T(this->measurement()(1));
        return true;
    }

    void linearizeOplus() override {
        auto camera_pose = static_cast<const VertexCameraBAL *>(vertex(0));
        auto point = static_cast<const VertexPointBAL *>(vertex(1));

        // jacs
        Eigen::Matrix<double, Dimension, VertexCameraBAL::Dimension, Eigen::RowMajor> dError_dCamera;
        Eigen::Matrix<double, Dimension, VertexPointBAL::Dimension, Eigen::RowMajor> dError_dPoint;
        double *jacobians[] = {dError_dCamera.data(), dError_dPoint.data()};

        // optimizing parameters
        double *parameters[] = {const_cast<double *>(camera_pose->estimate().data()),
                                const_cast<double *>(point->estimate().data())};

        // function value
        double value[Dimension];

        // auto diff
        typedef ceres::internal::AutoDiff<EdgeObservationBAL, double, VertexCameraBAL::Dimension, VertexPointBAL::Dimension> BalAutoDiff;
        bool diff_state = BalAutoDiff::Differentiate(
                *this,
                parameters,
                Dimension,
                value,
                jacobians
        );

        if (diff_state) {
            _jacobianOplusXi = dError_dCamera;
            _jacobianOplusXj = dError_dPoint;
        } else {
            _jacobianOplusXi.setZero();
            _jacobianOplusXj.setZero();
            assert(0 && "Calc auto diff error");
        }
    }

    void computeError() override {
        auto camera_pose = static_cast<const VertexCameraBAL *>(vertex(0));
        auto point = static_cast<const VertexPointBAL *>(vertex(1));

        // 计算loss已经在重载()里定义了，就直接使用了
        this->operator()(camera_pose->estimate().data(), point->estimate().data(), _error.data());

    }

    bool read(std::istream &is) override {
        return false;
    }

    bool write(std::ostream &os) const override {
        return false;
    }

};

void build_problem(const BALProblem *problem, g2o::SparseOptimizer *optimizer, const BundleParams_G2O &params) {
    const int num_points = problem->num_points();
    const int num_cameras = problem->num_cameras();
    const int camera_block_size = problem->camera_block_size();
    const int point_block_size = problem->point_block_size();

    // 添加camera vertex
    auto ptr_raw_cameras = problem->cameras();
    for (int vertex_id = 0; vertex_id < num_cameras; ++vertex_id) {
        // 这个map是c++ array在Eigen里的view
        Eigen::Map<const Eigen::VectorXd> temp_vec_camera(ptr_raw_cameras + camera_block_size * vertex_id,
                                                          camera_block_size);
        auto camera_pose = new VertexCameraBAL();

        camera_pose->setId(vertex_id);
        camera_pose->setEstimate(temp_vec_camera);
        optimizer->addVertex(camera_pose);
    }

    // 添加points vertex
    auto ptr_raw_points = problem->points();
    for (int vertex_id = 0; vertex_id < num_points; ++vertex_id) {
        Eigen::Map<const Eigen::VectorXd> tem_vec_point(ptr_raw_points + point_block_size * vertex_id,
                                                        point_block_size);
        auto point = new VertexPointBAL();

        point->setId(num_cameras + vertex_id);
        point->setEstimate(tem_vec_point);

        // 设置该点在解方程时进行 Schur 消元
        point->setMarginalized(true);
        optimizer->addVertex(point);
    }

    // 添加edges
    int num_observation = problem->num_observations();
    auto ptr_raw_observation = problem->observations();

    for (int edge_id = 0; edge_id < num_observation; ++edge_id) {
        auto edge_obs = new EdgeObservationBAL();

        // 当前这条边对应的camera和point的id
        int camera_pose_id = problem->camera_index()[edge_id];
        int point_id = problem->point_index()[edge_id] + num_cameras;

//        edge_obs->setId(edge_id);
        edge_obs->setVertex(0, dynamic_cast<VertexCameraBAL *>(optimizer->vertex(camera_pose_id)));
        edge_obs->setVertex(1, dynamic_cast<VertexPointBAL *>(optimizer->vertex(point_id)));

        edge_obs->setInformation(Eigen::Matrix2d::Identity());
        edge_obs->setMeasurement(Eigen::Vector2d(
                ptr_raw_observation[2 * edge_id + 0], ptr_raw_observation[2 * edge_id + 1]));

        // 鲁邦kernel
        if (params.robustify) {
            auto rk = new g2o::RobustKernelHuber();
            rk->setDelta(1.0);
            edge_obs->setRobustKernel(rk);
        }

        // add edge
        optimizer->addEdge(edge_obs);
    }
}

typedef g2o::BlockSolver<g2o::BlockSolverTraits<9, 3>> BALBlockSolver;

void set_g2o_optimize_flags(const BALProblem *problem, g2o::SparseOptimizer *optimizer, const BundleParams_G2O &params) {
    // 求解器
    g2o::LinearSolver<BALBlockSolver::PoseMatrixType> *linearSolver = 0;
    if (params.linear_solver == "dense_schur") {
        linearSolver = new g2o::LinearSolverDense<BALBlockSolver::PoseMatrixType>;
    }
    if (params.linear_solver == "sparse_schur") {
        linearSolver = new g2o::LinearSolverCholmod<BALBlockSolver::PoseMatrixType>;
        // 让 solver 对矩阵进行排序保持稀疏性
        dynamic_cast<g2o::LinearSolverCholmod<BALBlockSolver::PoseMatrixType> *>(linearSolver)->setBlockOrdering(true);
    }

    // 实际优化H阵的块求解器
    BALBlockSolver *solver_ptr;
    solver_ptr = new BALBlockSolver(linearSolver);

    // 优化方法
    g2o::OptimizationAlgorithmWithHessian *optimize_algorithm;
    if (params.trust_region_strategy == "levenberg_marquardt") {
        optimize_algorithm = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    } else if (params.trust_region_strategy == "dogleg") {
        optimize_algorithm = new g2o::OptimizationAlgorithmDogleg(solver_ptr);
    } else {
        assert(0 && "Unsupported optimizing algorithm.");
    }

    optimizer->setAlgorithm(optimize_algorithm);
}


void WriteToBALProblem(BALProblem *bal_problem, g2o::SparseOptimizer *optimizer) {
    const int num_points = bal_problem->num_points();
    const int num_cameras = bal_problem->num_cameras();
    const int camera_block_size = bal_problem->camera_block_size();
    const int point_block_size = bal_problem->point_block_size();

    double *raw_cameras = bal_problem->mutable_cameras();
    for (int i = 0; i < num_cameras; ++i) {
        VertexCameraBAL *pCamera = dynamic_cast<VertexCameraBAL *>(optimizer->vertex(i));
        Eigen::VectorXd NewCameraVec = pCamera->estimate();
        memcpy(raw_cameras + i * camera_block_size, NewCameraVec.data(), sizeof(double) * camera_block_size);
    }

    double *raw_points = bal_problem->mutable_points();
    for (int j = 0; j < num_points; ++j) {
        VertexPointBAL *pPoint = dynamic_cast<VertexPointBAL *>(optimizer->vertex(j + num_cameras));
        Eigen::Vector3d NewPointVec = pPoint->estimate();
        memcpy(raw_points + j * point_block_size, NewPointVec.data(), sizeof(double) * point_block_size);
    }
}

int main(int argc, char **argv) {
    BundleParams_G2O params(argc, argv);
    if (params.input.empty()) {
        std::cout << "Usage: bundle_adjuster -input <path for dataset>";
        return 1;
    }

    // 载入数据
    BALProblem problem(params.input);

    // show some information here ...
    std::cout << "bal problem file loaded..." << std::endl;
    std::cout << "bal problem have " << problem.num_cameras() << " cameras and "
              << problem.num_points() << " points. " << std::endl;
    std::cout << "Forming " << problem.num_observations() << " observatoins. " << std::endl;

    // store the initial 3D cloud points and camera pose..
    if (!params.initial_ply.empty()) {
        problem.WriteToPLYFile(params.initial_ply);
    }

    std::cout << "beginning problem..." << std::endl;

    // 预处理数据
    srand(params.random_seed);
    problem.Normalize();
    problem.Perturb(params.rotation_sigma, params.translation_sigma, params.point_sigma);
    std::cout << "Normalization complete..." << std::endl;

    // 配置优化器
    g2o::SparseOptimizer optimizer;
    set_g2o_optimize_flags(&problem, &optimizer, params);
    build_problem(&problem, &optimizer, params);

    // 开始求解
    std::cout << "begin optimizaiton .." << std::endl;
    optimizer.initializeOptimization();
    optimizer.setVerbose(true);
    optimizer.optimize(params.num_iterations);

    // 保存结果

    std::cout << "optimization complete.. " << std::endl;
    // write the optimized data into BALProblem class
    WriteToBALProblem(&problem, &optimizer);

    if (!params.final_ply.empty()) {
        problem.WriteToPLYFile(params.final_ply);
    }

    return 0;
}
