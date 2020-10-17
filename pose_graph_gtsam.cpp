//
// Created by hvt on 2020/10/12.
//

#include <gtsam/slam/dataset.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

#include <iostream>
#include <fstream>

using namespace std;

int main() {

    gtsam::NonlinearFactorGraph::shared_ptr graph(new gtsam::NonlinearFactorGraph);
    gtsam::Values::shared_ptr initial(new gtsam::Values);

    int vertex_count = 0, edge_count = 0;
    // 读取pose graph
//    auto pose_graph_path = "/home/hvt/Code/slambook/ch11/sphere.g2o";
    auto pose_graph_path = "/Users/hviktortsoi/Code/slambook/ch11/sphere.g2o";
    ifstream fin(pose_graph_path);
    while (!fin.eof()) {
        string type;
        fin >> type;
        cout << type << endl;
        if (type == "VERTEX_SE3:QUAT") {
            // 添加节点
            gtsam::Key vertex_id;
            fin >> vertex_id;
            double data[7];
            for (double &item : data) fin >> item;

            gtsam::Rot3 R = gtsam::Rot3::Quaternion(data[6], data[3], data[4], data[5]);
            gtsam::Point3 t(data[0], data[1], data[2]);
            initial->insert(vertex_id, gtsam::Pose3(R, t));
            vertex_count++;

        } else if (type == "EDGE_SE3:QUAT") {
            // 添加边, 对应到因子图中的因子
            gtsam::Key vert_i, vert_j;
            fin >> vert_i >> vert_j;

            // 读取相对位姿观测值
            double data[7];
            for (auto &item : data) fin >> item;
            gtsam::Rot3 R = gtsam::Rot3::Quaternion(data[6], data[3], data[4], data[5]);
            gtsam::Point3 t(data[0], data[1], data[2]);

            // 信息矩阵
            gtsam::Matrix info_mat_g2o = gtsam::I_6x6;
            for (int i = 0; i < 6; ++i) {
                for (int j = i; j < 6; ++j) {
                    double m_ij;
                    fin >> m_ij;
                    info_mat_g2o(i, j) = info_mat_g2o(j, i) = m_ij;
                }
            }
            // g2o 的信息矩阵定义方式与 gtsam 不同,这里对它进行修改
            gtsam::Matrix info_mat = gtsam::I_6x6;
            info_mat.block<3, 3>(0, 0) = info_mat_g2o.block<3, 3>(3, 3); // cov rotation
            info_mat.block<3, 3>(3, 3) = info_mat_g2o.block<3, 3>(0, 0); // cov translation
            info_mat.block<3, 3>(0, 3) = info_mat_g2o.block<3, 3>(0, 3); // off diagonal
            info_mat.block<3, 3>(3, 0) = info_mat_g2o.block<3, 3>(3, 0); // off diagonal

            // 高斯噪声模型
            gtsam::SharedNoiseModel model = gtsam::noiseModel::Gaussian::Information(info_mat);

            // 添加边因子
            gtsam::NonlinearFactor::shared_ptr factor(
                    new gtsam::BetweenFactor<gtsam::Pose3>(vert_i, vert_j, gtsam::Pose3(R, t), model)
            );
            graph->add(factor);
            edge_count++;

        } else {
            if (!fin.good()) {
                break;
            }
            assert(0 && "Unsupport data type!!");
        }

    }

    // 固定第一个顶点,在 gtsam 中相当于添加一个先验因子
    gtsam::NonlinearFactorGraph graph_with_prior = *graph;
    gtsam::noiseModel::Diagonal::shared_ptr priorModel = gtsam::noiseModel::Diagonal::Variances(
            (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6).finished());

    gtsam::Key firstKey = 0;
    for (const gtsam::Values::ConstKeyValuePair &key_value:*initial) {
        graph_with_prior.add(gtsam::PriorFactor<gtsam::Pose3>(
                key_value.key, key_value.value.cast<gtsam::Pose3>(), priorModel
        ));
        // 只对第一个节点添加先验因子
        break;
    }
    cout << "Prepare optimizing..." << endl;
    gtsam::LevenbergMarquardtParams params_lm;
    params_lm.setVerbosity("ERROR");
    params_lm.setMaxIterations(20);
    params_lm.setLinearSolverType("MULTIFRONTAL_QR");
    gtsam::LevenbergMarquardtOptimizer optimizer(graph_with_prior, *initial, params_lm);

    gtsam::Values result = optimizer.optimize();
    cout << "Optimization complete" << endl;
    cout << "initial error: " << graph->error(*initial) << endl;
    cout << "final error: " << graph->error(result) << endl;

    ofstream fout("result_gtsam.g2o");
    ofstream csv_out("result_gtsam.csv");
    for (const gtsam::Values::KeyValuePair key_value:result) {
        gtsam::Pose3 solved_pose = key_value.value.cast<gtsam::Pose3>();
        auto t = solved_pose.translation();
        auto R = solved_pose.rotation().quaternion();
        fout << "VERTEX_SE3:QUAT " << key_value.key
             << " " << t.x() << " " << t.y() << " " << t.z() << " "
             << R.x() << " " << R.y() << " " << R.z() << " " << R.w() << " " << endl;
        csv_out << t.x() << ", " << t.y() << ", " << t.z() << endl;
    }
    // edges
    for (auto &factor:*graph) {
        gtsam::BetweenFactor<gtsam::Pose3>::shared_ptr between_factor =
                dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3>>(factor);
        if (between_factor) {
            gtsam::SharedNoiseModel model = between_factor->noiseModel();
            gtsam::noiseModel::Gaussian::shared_ptr gausissian_model =
                    dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(model);
            if (gausissian_model) {
                gtsam::Matrix info_mat = gausissian_model->R().transpose() * gausissian_model->R();
                gtsam::Pose3 solved_pose = between_factor->measured();
                auto t = solved_pose.translation();
                auto R = solved_pose.rotation().toQuaternion();

                fout << "EDGE_SE3:QUAT " << between_factor->key1() << " " << between_factor->key2()
                     << " " << t.x() << " " << t.y() << " " << t.z() << " "
                     << R.x() << " " << R.y() << " " << R.z() << " " << R.w() << " ";

                gtsam::Matrix info_g2o = gtsam::I_6x6;
                info_g2o.block(0, 0, 3, 3) = info_mat.block(3, 3, 3, 3); // cov translation
                info_g2o.block(3, 3, 3, 3) = info_mat.block(0, 0, 3, 3); // cov rotation
                info_g2o.block(0, 3, 3, 3) = info_mat.block(0, 3, 3, 3); // off diagonal
                info_g2o.block(3, 0, 3, 3) = info_mat.block(3, 0, 3, 3); // off diagonal
                for (int i = 0; i < 6; ++i) {
                    for (int j = 0; j < 6; ++j) {
                        fout << info_g2o(i, j) << " ";
                    }
                }
                fout << endl;
            }
        }
    }
    fout.close();
    csv_out.close();
    return 0;

}