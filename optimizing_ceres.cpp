#include <iostream>

#include <ceres/ceres.h>
#include <ceres/internal/autodiff.h>
#include "common/projection.h"
#include "common/BALProblem.h"
#include "common/BundleParams_Ceres.h"

class SnavelyReprojectionError {
public:
    SnavelyReprojectionError(double observedX, double observedY)
            : observed_x(observedX), observed_y(observedY) {}

    template<typename T>
    bool operator()(const T *const camera, const T *const point, T *residual) const {
        T prediction[2];
        // 利用估计的位姿和估计的点 计算投影在估计像平面的点
        CamProjectionWithDistortion(camera, point, prediction);
        residual[0] = prediction[0] - T(observed_x);
        residual[1] = prediction[1] - T(observed_y);
        return true;
    }

    static ceres::CostFunction *Create(const double observation_x, const double observation_y) {
        return new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
                new SnavelyReprojectionError(observation_x, observation_y));
    }

private:
    double observed_x;
    double observed_y;
};

void build_problem(BALProblem *bal_problem, ceres::Problem *problem, BundleParams_Ceres &params) {
    const int camera_block_size = bal_problem->camera_block_size();
    const int point_block_size = bal_problem->point_block_size();

    double *raw_camera_poses = bal_problem->mutable_cameras();
    double *raw_points = bal_problem->mutable_points();

    const double *raw_observations = bal_problem->observations();

    // 构建问题
    for (int i = 0; i < bal_problem->num_observations(); ++i) {
        // cost function
        ceres::CostFunction *cost_function;
        cost_function = SnavelyReprojectionError::Create(
                raw_observations[2 * i + 0], raw_observations[2 * i + 1]);

        // 鲁棒kernel
        ceres::LossFunction *kernel_function = params.robustify ? new ceres::HuberLoss(1.0) : NULL;

        double *var_camera_pose = raw_camera_poses + camera_block_size * bal_problem->camera_index()[i];
        double *var_point = raw_points + point_block_size * bal_problem->point_index()[i];

        problem->AddResidualBlock(cost_function, kernel_function, var_camera_pose, var_point);
    }
}

void set_ordering(BALProblem *bal_problem, ceres::Solver::Options *options, const BundleParams_Ceres &params) {
    const int camera_block_size = bal_problem->camera_block_size();
    const int point_block_size = bal_problem->point_block_size();

    double *raw_camera_poses = bal_problem->mutable_cameras();
    double *raw_points = bal_problem->mutable_points();

    ceres::ParameterBlockOrdering *ordering = new ceres::ParameterBlockOrdering;

    // 相机位姿1组
    for (int i = 0; i < bal_problem->num_cameras(); ++i) {
        ordering->AddElementToGroup(raw_camera_poses + camera_block_size * i, 1);
    }
    // 路标点0组 先进行消元
    for (int i = 0; i < bal_problem->num_points(); ++i) {
        ordering->AddElementToGroup(raw_points + point_block_size * i, 0);
    }

    options->linear_solver_ordering.reset(ordering);
}

void set_optimizer_flags(BALProblem *bal_problem, ceres::Solver::Options *options, BundleParams_Ceres &params) {
    options->max_num_iterations = params.num_iterations;
    options->minimizer_progress_to_stdout = true;
    options->num_threads = params.num_threads;

    CHECK(ceres::StringToTrustRegionStrategyType(params.trust_region_strategy,
                                                 &options->trust_region_strategy_type));
    CHECK(ceres::StringToLinearSolverType(params.linear_solver,
                                          &options->linear_solver_type));
    CHECK(ceres::StringToSparseLinearAlgebraLibraryType(params.sparse_linear_algebra_library,
                                                        &options->sparse_linear_algebra_library_type));
    CHECK(ceres::StringToDenseLinearAlgebraLibraryType(params.dense_linear_algebra_library,
                                                       &options->dense_linear_algebra_library_type));

    options->gradient_tolerance = 1e-16;
    options->function_tolerance = 1e-16;

    set_ordering(bal_problem, options, params);
}

int main(int argc, char **argv) {
    BundleParams_Ceres params(argc, argv);
    if (params.input.empty()) {
        std::cout << "Usage: bundle_adjuster -input <path for dataset>";
        return 1;
    }

    // 载入数据
    BALProblem bal_problem(params.input);

    // show some information here ...
    std::cout << "bal problem file loaded..." << std::endl;
    std::cout << "bal problem have " << bal_problem.num_cameras() << " cameras and "
              << bal_problem.num_points() << " points. " << std::endl;
    std::cout << "Forming " << bal_problem.num_observations() << " observatoins. " << std::endl;

    // store the initial 3D cloud points and camera pose..
    if (!params.initial_ply.empty()) {
        bal_problem.WriteToPLYFile(params.initial_ply);
    }

    std::cout << "beginning problem..." << std::endl;

    // 预处理数据
    srand(params.random_seed);
    bal_problem.Normalize();
    bal_problem.Perturb(params.rotation_sigma, params.translation_sigma, params.point_sigma);
    std::cout << "Normalization complete..." << std::endl;

    // 构建优化问题
    ceres::Problem problem;
    build_problem(&bal_problem, &problem, params);

    // 求解器
    ceres::Solver::Options options;
    set_optimizer_flags(&bal_problem, &options, params);

    // log
    ceres::Solver::Summary summary;

    // 求解
    ceres::Solve(options, &problem, &summary);

    // ！！！！ 注意 ceres就没有把数据复制到optimizer里 直接在原始内存里原地优化！！
    if (!params.final_ply.empty()) {
        bal_problem.WriteToPLYFile(params.final_ply);
    }

    return 0;
}
