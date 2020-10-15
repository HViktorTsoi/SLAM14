//
// Created by hvt on 2020/10/12.
//

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <iostream>
#include <fstream>

using namespace std;

int main() {
    g2o::SparseOptimizer optimizer;

    // solver, J is 6x6
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> Block;
    Block::LinearSolverType *linearSolver = new g2o::LinearSolverCholmod<Block::PoseMatrixType>();
    Block *solver_ptr = new Block(linearSolver);
    auto algorithm = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(algorithm);

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

            auto *new_vertex = new g2o::VertexSE3();
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

            auto *new_edge = new g2o::EdgeSE3();
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

    optimizer.save("result.g2o");
    return 0;
}