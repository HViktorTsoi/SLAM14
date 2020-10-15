//
// Created by hvt on 2020/10/12.
//

#include <gtsam/slam/dataset.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

#include <sophus/se3.h>
#include <sophus/so3.h>

#include <iostream>
#include <fstream>

using namespace std;

int main() {

    gtsam::NonlinearFactorGraph::shared_ptr graph(new gtsam::NonlinearFactorGraph);
    gtsam::Values::shared_ptr initial(new gtsam::Values);

    int vertex_count = 0, edge_count = 0;
    // 读取pose graph
    auto pose_graph_path = "/home/hvt/Code/slambook/ch11/sphere.g2o";
    ifstream fin(pose_graph_path);
    while (!fin.eof()) {
        string type;
        fin >> type;
        cout << type << endl;
        if (type == "VERTEX_SE3:QUAT") {
            // 添加节点
            int vertex_id;
            fin >> vertex_id;
            vertex_count++;

            auto *new_vertex = new VertexSE3LieAlgebra();
            new_vertex->setId(vertex_id);
            new_vertex->read(fin);
            // 第一个点的位姿不进行优化
            if (vertex_id == 0) {
                new_vertex->setFixed(true);
            }

            optimizer.addVertex(new_vertex);

        } else if (type == "EDGE_SE3:QUAT") {
            // 添加边
            int vert_i, vert_j;
            fin >> vert_i >> vert_j;

            auto *new_edge = new EdgeSE3LieAlgebra();
            new_edge->setId(edge_count);
            new_edge->setVertex(0, optimizer.vertices()[vert_i]);
            new_edge->setVertex(1, optimizer.vertices()[vert_j]);
            new_edge->read(fin);

            optimizer.addEdge(new_edge);

        } else {
            if (!fin.good()) {
                break;
            }
            assert(0 && "Unsupport data type!!");
        }

    }

    cout << "Prepare optimizing..." << endl;

    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(30);

    // save result
    ofstream fout("result_lie.g2o");
    for (auto &p:optimizer.vertices()) {
        fout << "VERTEX_SE3:QUAT" << " ";
        static_cast<VertexSE3LieAlgebra *>(p.second)->write(fout);
    }
    for (auto &p:optimizer.edges()) {
        fout << "EDGE_SE3:QUAT" << " ";
        static_cast<EdgeSE3LieAlgebra *>(p)->write(fout);
    }
    return 0;
}