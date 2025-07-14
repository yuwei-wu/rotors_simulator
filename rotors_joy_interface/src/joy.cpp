/*
 * Copyright 2015 Fadri Furrer, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Michael Burri, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Mina Kamel, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Janosch Nikolic, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Markus Achtelik, ASL, ETH Zurich, Switzerland
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
//out put way point

#include "rotors_joy_interface/joy.h"
#include "mav_msgs/eigen_mav_msgs.h"

#include <mav_msgs/default_topics.h>

#include <Eigen/Core>
#include <mav_msgs/conversions.h>
// #include <mav_msgs/default_topics.h>
// #include <ros/ros.h>
// #include <std_srvs/Empty.h>
#include <trajectory_msgs/MultiDOFJointTrajectory.h>

#include <gazebo_msgs/ApplyBodyWrench.h>
#include <thread>
#include <chrono>
#include <std_srvs/Empty.h>
#include <geometry_msgs/Point.h>

int autoflight = 1;
#include <visualization_msgs/Marker.h>
visualization_msgs::Marker target;
void MarkerShowInit()
{
  target.header.frame_id = "world";
  target.ns = "target";
  target.action = visualization_msgs::Marker::ADD;
  target.pose.orientation.w = 1.0;
  target.type = visualization_msgs::Marker::POINTS;

  target.id = 0;
  target.scale.x = 0.3;
  target.scale.y = 0.3;
  target.color.a = 1;
  target.color.g = 1;
}

#include "std_msgs/Int8.h"


ros::ServiceClient client;

ros::Publisher trajectory_pub; 
ros::Publisher disturbance_pub;
ros::Subscriber pose_sub;

ros::Publisher marker_rviz_pub;

Eigen::Vector3d position_gt;
double yaw_gt;
double PI = 3.1415926;

bool get_position_gt = false;



void gt_callback(const nav_msgs::Odometry::ConstPtr &msg)
{
    position_gt[0] = msg->pose.pose.position.x;
    position_gt[1] = msg->pose.pose.position.y;
    position_gt[2] = msg->pose.pose.position.z;


    Eigen::Quaterniond quaternion;
    quaternion.w() = msg->pose.pose.orientation.w;
    quaternion.x() = msg->pose.pose.orientation.x;
    quaternion.y() = msg->pose.pose.orientation.y;
    quaternion.z() = msg->pose.pose.orientation.z;

  
    Eigen::Vector3d temp2;
    mav_msgs::getEulerAnglesFromQuaternion(quaternion, &temp2);

    yaw_gt =  temp2(2);

    get_position_gt = true;


}
int N = 20;
double R = 2, V = 1;
double dis_min = 1;

Joy::Joy() {
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  trajectory_pub =
      nh_.advertise<trajectory_msgs::MultiDOFJointTrajectory>(
          mav_msgs::default_topics::COMMAND_TRAJECTORY, 10);

  disturbance_pub = nh_.advertise<std_msgs::Int8>("disturbance/cmd", 10);

  pose_sub = nh_.subscribe("/hummingbird/ground_truth/odometry", 100, gt_callback, ros::TransportHints().tcpNoDelay()); //identification


  pnh.param("autoflight", autoflight, 0);

  pnh.param("velocity", V, 1.0);
  pnh.param("Num", N, 20);
  pnh.param("Radius", R, 2.0);
  pnh.param("dis_min", dis_min, 1.0);
  
  pnh.param("axis_roll_", axes_.roll, 0);
  pnh.param("axis_pitch_", axes_.pitch, 1);
  pnh.param("axis_thrust_", axes_.thrust, 2);

  pnh.param("axis_direction_roll", axes_.roll_direction, -1);
  pnh.param("axis_direction_pitch", axes_.pitch_direction, 1);
  pnh.param("axis_direction_thrust", axes_.thrust_direction, 1);

  pnh.param("v_yaw_step", v_yaw_step_, 0.05);  // [rad/s]

  pnh.param("button_yaw_left_", buttons_.yaw_left, 3);
  pnh.param("button_yaw_right_", buttons_.yaw_right, 4);
  pnh.param("button_ctrl_enable_", buttons_.ctrl_enable, 5);
  pnh.param("button_ctrl_mode_", buttons_.ctrl_mode, 10);
  pnh.param("button_takeoff_", buttons_.takeoff, 7);
  pnh.param("button_land_", buttons_.land, 8);

  /*
  botton: 
  lf:4 rf:5 Y3 B1 X2 A0
   */


  namespace_ = nh_.getNamespace();
  joy_sub_ = nh_.subscribe("joy", 1000, &Joy::JoyCallback, this);
  client = nh_.serviceClient<gazebo_msgs::ApplyBodyWrench>("/gazebo/apply_body_wrench");

}


void Joy::JoyCallback(const sensor_msgs::JoyConstPtr& msg) {
  current_joy_ = *msg;
  // std::cout << "joy " << std::endl;
  static double x_,  y_, z_ , yaw_;
  static bool buttons_up_midyaw_p = true, buttons_up_midyaw_n = true;
  
  if (!get_position_gt) {
    std::cout << "waiting for ground truth position..." << std::endl;
    return;
  }

  x_ = position_gt[0] + msg->axes[0]*1.0;
  y_ = position_gt[1] + msg->axes[1]*1.0;
  z_ = position_gt[2] + msg->axes[2]*1.0;

  // if(msg->buttons[my_button.mid_yaw_p])  
  // {
  //   if(buttons_up_midyaw_p)
  //   {
  //     yaw0 += 30 * M_PI / 180.0; 
  //     buttons_up_midyaw_p = false;
  //   // yaw0 =  yaw0>120*M_PI/180.0 ? 120*M_PI/180.0 : yaw0; 
  //   }
  // }
  // else
  // {
  //   buttons_up_midyaw_p = true;
  // }
  // if(msg->buttons[my_button.mid_yaw_n])  
  // {
  //   if(buttons_up_midyaw_n)
  //   {
  //     yaw0 -= 30 * M_PI / 180.0; 
  //     // yaw0 =  yaw0<-120*M_PI/180.0 ? -120*M_PI/180.0 : yaw0;
  //     buttons_up_midyaw_n = false;
  //   }
  // }
  // else
  // {
  //   buttons_up_midyaw_n = true;
  // }
  
  if (yaw_gt  +  msg->axes[6]*PI > PI){
    yaw_ = PI;
  }

  if (yaw_gt  +  msg->axes[6]*PI < -PI){
    yaw_ = -PI;
  }

  yaw_ =  yaw_gt  +  msg->axes[6]*3.1415926;

  // std::cout << "T: " << msg->axes[my_axes.thrust] << " " << msg->axes[my_axes.pitch] << " " << msg->axes[my_axes.roll] << " " << msg->axes[my_axes.yaw] << std::endl;
  // std::cout << "B: " << msg->buttons[0] << msg->buttons[1] << msg->buttons[2] << msg->buttons[3] << msg->buttons[4] << msg->buttons[5] << msg->buttons[6] << msg->buttons[7] << msg->buttons[8] << msg->buttons[9] << std::endl;

  gazebo_msgs::ApplyBodyWrench srv;
  srv.request.body_name = "hummingbird::hummingbird/base_link";
  geometry_msgs::Point P;
  P.x = 0.0;
  P.y = 0.0;
  P.z = 0.0;
  srv.request.reference_point = P;
  // srv.request.start_time = ros::Time;
  srv.request.wrench.force.x = msg->axes[3];//10  //world frame
  srv.request.wrench.force.y = msg->axes[4];
  srv.request.wrench.force.z = msg->axes[5];
  srv.request.wrench.torque.x = 0.0;  //0.4//axis is in world axis but center in body
  srv.request.wrench.torque.y = 0.0;
  srv.request.wrench.torque.z = 0.0;
  srv.request.duration.sec = 10;//100000000;
  srv.request.reference_frame = "world"; //world(inertial) hummingbird/base_link same result ？？？？TODO change the force and torque to body frame  //now it is world frame direction at the body point

  client.call(srv);


  // if(msg->buttons[my_button.force]) //Y
  // {
  //   #if 0
  //   gazebo_msgs::ApplyBodyWrench srv;
  //   srv.request.body_name = "hummingbird::hummingbird/base_link";
  //   geometry_msgs::Point P;
  //   P.x = 0.0;
  //   P.y = 0.0;
  //   P.z = 0.0;
  //   srv.request.reference_point = P;
  //   // srv.request.start_time = ros::Time;
  //   srv.request.wrench.force.x = 10.0;//10  //world frame
  //   srv.request.wrench.force.y = 0.0;
  //   srv.request.wrench.force.z = 0.0;
  //   srv.request.wrench.torque.x = 0.0;  //0.4//axis is in world axis but center in body
  //   srv.request.wrench.torque.y = 0.0;
  //   srv.request.wrench.torque.z = 0.0;
  //   srv.request.duration.sec = 10;//100000000;
  //   // srv.request.reference_frame = "world"; //world(inertial) hummingbird/base_link same result ？？？？TODO change the force and torque to body frame  //now it is world frame direction at the body point

  //   client.call(srv);
  //   // bool unpaused = ros::service::call("/gazebo/apply_body_wrench '{body_name: "hummingbird::hummingbird/base_link",reference_point: {x: 0.0, y: 0.0, z: 0.0}, wrench:{ force: {x: 10, y: 0.0, z: 0.0}}, start_time: 0, duration: 1000000000}'", srv);
  //   #endif
   
  //   // if(unpaused)
  //   std_msgs::Int8 disturbance_cmd;
  //   disturbance_cmd.data = 1;
  //   disturbance_pub.publish(disturbance_cmd);
  //     // ROS_INFO("apply force ! ");
  //   // else
  //   //   ROS_INFO("apply force fail! ");
  // }

  if(!autoflight)
  {
    trajectory_msgs::MultiDOFJointTrajectory trajectory_msg;
    trajectory_msg.header.stamp = ros::Time::now();

    // Default desired position and yaw.
    Eigen::Vector3d desired_position(x_, y_, z_);
    double desired_yaw = yaw_;
    // Eigen::Vector3d desired_position(0.0, 0.0, 1.0);
    // double desired_yaw = 0.0;

    mav_msgs::msgMultiDofJointTrajectoryFromPositionYaw(
        desired_position, desired_yaw, &trajectory_msg);
    trajectory_pub.publish(trajectory_msg);  
  }
  

  std::cout << "joy" << std::endl; 
}

int main(int argc, char** argv) {

  ros::init(argc, argv, "rotors_joy_interface");
  Joy joy;

  ros::spin();


  return 0;
}