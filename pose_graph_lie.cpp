//
// Created by hvt on 2020/10/12.
//

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <sophus/se3.h>
#include <sophus/so3.h>

#include <iostream>
#include <fstream>

using namespace std;

typedef Eigen::Matrix<double, 6, 6> Matrix6d;

class VertexSE3LieAlgebra : public g2o::BaseVertex<6, Sophus::SE3> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bool read(istream &is) override {
        double data[7];
        for (int i = 0; i < 7; ++i) {
            is >> data[i];
        }
        this->setEstimate(Sophus::SE3(
                Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
                Eigen::Vector3d(data[0], data[1], data[2])
        ));
        return true;
    }

    bool write(ostream &os) const override {
        // 保存
        os << this->id() << " ";
        Eigen::Quaterniond q = this->estimate().unit_quaternion();
        os << this->estimate().translation().transpose() << " ";
        os << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3];
        os << endl;
        return true;
    }

protected:
    void oplusImpl(const double *update_se3) override {
        // 增量更新
        Sophus::SE3 update_SE3(
                Sophus::SO3(update_se3[3], update_se3[4], update_se3[5]),
                Eigen::Vector3d(update_se3[0], update_se3[1], update_se3[2])
        );
        this->_estimate = update_SE3 * this->estimate();
    }

    void setToOriginImpl() override {
        this->_estimate = Sophus::SE3();
    }
};

class EdgeSE3LieAlgebra : public g2o::BaseBinaryEdge<6, Sophus::SE3, VertexSE3LieAlgebra, VertexSE3LieAlgebra> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void linearizeOplus() override {
        Sophus::SE3 v_from = static_cast<VertexSE3LieAlgebra *>(this->vertices()[0])->estimate();
        Sophus::SE3 v_to = static_cast<VertexSE3LieAlgebra *>(this->vertices()[1])->estimate();

        Matrix6d J = this->Jr_inv(Sophus::SE3::exp(this->error()));
        _jacobianOplusXi = -J * v_to.inverse().Adj();
        _jacobianOplusXj = J * v_to.inverse().Adj();
    }

    void computeError() override {
        Sophus::SE3 v_from = static_cast<VertexSE3LieAlgebra *>(this->vertices()[0])->estimate();
        Sophus::SE3 v_to = static_cast<VertexSE3LieAlgebra *>(this->vertices()[1])->estimate();

        // measurement是相对位姿(匹配得到) from to是待估计绝对位姿
        this->error() = (this->measurement().inverse() * v_from.inverse() * v_to).log();
    }

    bool read(istream &is) override {
        double data[7];
        for (int i = 0; i < 7; ++i) {
            is >> data[i];
        }
        Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);
        q.normalize();
        this->setMeasurement(Sophus::SE3(q, Eigen::Vector3d(data[0], data[1], data[2])));
        // information
        for (int i = 0; i < information().rows() && is.good(); i++)
            for (int j = i; j < information().cols() && is.good(); j++) {
                is >> this->information()(i, j);
                if (i != j)
                    this->information()(j, i) = this->information()(i, j);
            }
        return true;
    }

    bool write(ostream &os) const override {
        auto v_from = static_cast<VertexSE3LieAlgebra *>(this->vertices()[0]);
        auto v_to = static_cast<VertexSE3LieAlgebra *>(this->vertices()[1]);

        os << v_from->id() << " " << v_to->id();
        os << this->measurement().translation().transpose() << " ";
        auto q = this->measurement().unit_quaternion();
        os << this->measurement().translation().transpose() << " ";
        os << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << " ";

        // information matrix
        for (int i = 0; i < information().rows(); i++)
            for (int j = i; j < information().cols(); j++) {
                os << information()(i, j) << " ";
            }
        os << endl;

        return true;
    }

private:
    Matrix6d Jr_inv(Sophus::SE3 error) {
        Matrix6d Jr;
        Jr.block(0, 0, 3, 3) = Sophus::SO3::hat(error.so3().log());
        Jr.block(0, 3, 3, 3) = Sophus::SO3::hat(error.translation());
        Jr.block(3, 0, 3, 3) = Eigen::Matrix3d::Zero();
        Jr.block(3, 3, 3, 3) = Sophus::SO3::hat(error.so3().log());
        Jr = Matrix6d::Identity() + 0.5 * Jr;
        return Jr;
//        return Matrix6d::Identity();
    }
};


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