cmake_minimum_required(VERSION 2.8.3)
project(pose_utils)

find_package(catkin REQUIRED COMPONENTS
  #armadillo
  roscpp
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES pose_utils
#  CATKIN_DEPENDS geometry_msgs nav_msgs
#  DEPENDS system_lib
)

find_package(Armadillo REQUIRED)

include_directories(
    ${catkin_INCLUDE_DIRS}
    ${ARMADILLO_INCLUDE_DIRS}
    include
    )

add_library(pose_utils 
   src/pose_utils.cpp)

target_link_libraries(pose_utils
   ${catkin_LIBRARIES}
   ${ARMADILLO_LIBRARIES}
)
