/*
 * Copyright 2025 
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include <ros/ros.h>
#include <trajectory_msgs/MultiDOFJointTrajectory.h>
#include <quadrotor_msgs/PositionCommand.h>
#include <mav_msgs/conversions.h>
#include <mav_msgs/eigen_mav_msgs.h>
#include <mav_msgs/default_topics.h>

class EgoPlannerBridge {
public:
  EgoPlannerBridge() {
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");

    // Subscribe to ego-planner output
    position_cmd_sub_ = nh.subscribe("/planning/pos_cmd", 10, 
                                     &EgoPlannerBridge::positionCmdCallback, this);

    // Publish to rotors controller
    trajectory_pub_ = nh.advertise<trajectory_msgs::MultiDOFJointTrajectory>(
        mav_msgs::default_topics::COMMAND_TRAJECTORY, 10);

    ROS_INFO("Ego-planner bridge node started");
    ROS_INFO("  Subscribing to: /planning/pos_cmd");
    ROS_INFO("  Publishing to: %s", mav_msgs::default_topics::COMMAND_TRAJECTORY);
  }

  void positionCmdCallback(const quadrotor_msgs::PositionCommand::ConstPtr& msg) {
    // Convert ego-planner command to rotors trajectory message
    trajectory_msgs::MultiDOFJointTrajectory trajectory_msg;
    trajectory_msg.header = msg->header;

    // Create trajectory point
    Eigen::Vector3d desired_position(msg->position.x, msg->position.y, msg->position.z);
    Eigen::Vector3d desired_velocity(msg->velocity.x, msg->velocity.y, msg->velocity.z);
    Eigen::Vector3d desired_acceleration(msg->acceleration.x, msg->acceleration.y, msg->acceleration.z);
    double desired_yaw = msg->yaw;
    double desired_yaw_rate = msg->yaw_dot;

    // Use mav_msgs helper function
    mav_msgs::msgMultiDofJointTrajectoryFromPositionYaw(
        desired_position, desired_yaw, &trajectory_msg);

    // Add velocity and acceleration if needed
    if (trajectory_msg.points.size() > 0) {
      trajectory_msg.points[0].velocities.resize(1);
      trajectory_msg.points[0].velocities[0].linear.x = desired_velocity.x();
      trajectory_msg.points[0].velocities[0].linear.y = desired_velocity.y();
      trajectory_msg.points[0].velocities[0].linear.z = desired_velocity.z();
      trajectory_msg.points[0].velocities[0].angular.z = desired_yaw_rate;

      trajectory_msg.points[0].accelerations.resize(1);
      trajectory_msg.points[0].accelerations[0].linear.x = desired_acceleration.x();
      trajectory_msg.points[0].accelerations[0].linear.y = desired_acceleration.y();
      trajectory_msg.points[0].accelerations[0].linear.z = desired_acceleration.z();
    }

    trajectory_pub_.publish(trajectory_msg);
  }

private:
  ros::Subscriber position_cmd_sub_;
  ros::Publisher trajectory_pub_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "ego_planner_bridge");
  EgoPlannerBridge bridge;
  ros::spin();
  return 0;
}
