#include "rotors_gazebo_plugins/moving_target_plugin.h"
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <gazebo/common/Time.hh>
#include <gazebo/physics/physics.hh>
#include <ignition/math/Vector3.hh>

namespace gazebo
{
  void MovingTargetPlugin::Load(physics::ModelPtr model, sdf::ElementPtr sdf)
  {
    this->model = model;
    this->velocity = ignition::math::Vector3d(0.0, 0.0, 0.0);

    if (sdf->HasElement("velocity")) {
        this->velocity = sdf->Get<ignition::math::Vector3d>("velocity");
    }

    if (!ros::isInitialized()) {
      ROS_FATAL_STREAM("A ROS node for Gazebo has not been initialized, unable to load plugin.");
      return;
    }

    this->rosNode.reset(new ros::NodeHandle(""));

    // Subscribe to /move_base_simple/goal
    this->goalSub = this->rosNode->subscribe("/move_base_simple/goal", 1, &MovingTargetPlugin::OnGoalReceived, this);

    // Odometry Publisher
    std::string odomTopic = "/" + this->model->GetName() + "/odometry";
    this->odomPub = this->rosNode->advertise<nav_msgs::Odometry>(odomTopic, 10);

    // Connect to the simulation update loop
    this->updateConnection = event::Events::ConnectWorldUpdateBegin(
        std::bind(&MovingTargetPlugin::OnUpdate, this));

    this->lastUpdateTime = this->model->GetWorld()->SimTime();

    gzdbg << "MovingTargetPlugin loaded for model [" << model->GetName() << "] with velocity: " 
          << this->velocity << std::endl;
  }

  void MovingTargetPlugin::OnGoalReceived(const geometry_msgs::PoseStamped::ConstPtr& msg)
  {
    this->isMoving = true;
    gzdbg << "Received /move_base_simple/goal trigger. Target will start moving." << std::endl;
  }

  void MovingTargetPlugin::OnUpdate()
  {
    common::Time currentTime = this->model->GetWorld()->SimTime();
    double timeDelta = (currentTime - this->lastUpdateTime).Double();

    ignition::math::Pose3d currentPose = this->model->WorldPose();

    if (timeDelta > 0 && this->isMoving)
    {
      // Update position based on velocity
      ignition::math::Vector3d newPosition = currentPose.Pos() + this->velocity * timeDelta;
      this->model->SetWorldPose(ignition::math::Pose3d(newPosition, currentPose.Rot()));
      currentPose = this->model->WorldPose();  // Update currentPose after moving
    }

    this->lastUpdateTime = currentTime;

    // Always publish odometry
    this->PublishOdometry(currentPose);
  }

  void MovingTargetPlugin::PublishOdometry(const ignition::math::Pose3d& pose)
  {
    nav_msgs::Odometry odom;
    odom.header.stamp = ros::Time::now();
    odom.header.frame_id = "world";
    odom.child_frame_id = this->model->GetName();

    // Pose
    odom.pose.pose.position.x = pose.Pos().X();
    odom.pose.pose.position.y = pose.Pos().Y();
    odom.pose.pose.position.z = pose.Pos().Z();

    odom.pose.pose.orientation.x = pose.Rot().X();
    odom.pose.pose.orientation.y = pose.Rot().Y();
    odom.pose.pose.orientation.z = pose.Rot().Z();
    odom.pose.pose.orientation.w = pose.Rot().W();

    // Velocity
    if (this->isMoving)
    {
      odom.twist.twist.linear.x = this->velocity.X();
      odom.twist.twist.linear.y = this->velocity.Y();
      odom.twist.twist.linear.z = this->velocity.Z();
    }
    else
    {
      odom.twist.twist.linear.x = 0.0;
      odom.twist.twist.linear.y = 0.0;
      odom.twist.twist.linear.z = 0.0;
    }

    // Publish
    this->odomPub.publish(odom);
  }

  GZ_REGISTER_MODEL_PLUGIN(MovingTargetPlugin);
}
