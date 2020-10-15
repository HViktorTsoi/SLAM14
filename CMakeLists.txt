cmake_minimum_required(VERSION 3.15)
project(SLAM14)

set(CMAKE_CXX_STANDARD 14)
set(G2O_LIBS g2o_core g2o_types_sba g2o_solver_cholmod g2o_solver_csparse g2o_stuff g2o_csparse_extension)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(G2O REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Cholmod REQUIRED)
find_package(Sophus REQUIRED)
find_package(GTSAMCMakeTools)
find_package(GTSAM REQUIRED)

include_directories(${G2O_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${CHOLMOD_INCLUDE_DIR})
include_directories(${Sophus_INCLUDE_DIRS})
include_directories(${GTSAM_INCLUDE_DIR})

include_directories(common/flags)

message("" ${G2O_LIBS})

add_executable(G2O_OPTIMIZING optimizing_g2o.cpp common/BALProblem.cpp common/flags/command_args.cpp)

target_link_libraries(G2O_OPTIMIZING ${G2O_LIBS} ${CERES_LIBRARIES})

add_executable(CERES_OPTIMIZING optimizing_ceres.cpp common/BALProblem.cpp common/flags/command_args.cpp)

target_link_libraries(CERES_OPTIMIZING ${CERES_LIBRARIES})

add_executable(POSE_GRAPH_G2OSE3 pose_graph_g2o_se3.cpp)

target_link_libraries(POSE_GRAPH_G2OSE3 g2o_core g2o_stuff g2o_types_slam3d ${CHOLMOD_LIBRARIES})

add_executable(POSE_GRAPH_LIE pose_graph_lie.cpp)

target_link_libraries(POSE_GRAPH_LIE g2o_core g2o_stuff g2o_types_slam3d ${CHOLMOD_LIBRARIES} ${Sophus_LIBRARIES})

add_executable(POSE_GRAPH_GTSAM pose_graph_gtsam.cpp)

target_link_libraries(POSE_GRAPH_GTSAM ${CHOLMOD_LIBRARIES} ${Sophus_LIBRARIES} gtsam)