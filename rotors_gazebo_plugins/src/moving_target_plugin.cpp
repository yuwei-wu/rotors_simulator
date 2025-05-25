#include "rotors_gazebo_plugins/moving_target_plugin.h"

#include <nav_msgs/Odometry.h>
#include <ros/ros.h>

namespace gazebo
{
  void MovingTargetPlugin::Load(physics::ModelPtr model, sdf::ElementPtr sdf)
  {
    this->model = model;
    this->velocity = ignition::math::Vector3d(0, 0, 0);

    // Load velocity from SDF
    if (sdf->HasElement("velocity")) {
        this->velocity = sdf->Get<ignition::math::Vector3d>("velocity");
    } else {
        gzerr << "Missing <velocity> in SDF. Plugin won't move.\n";
        return;
    }

    // Load topic from SDF
    std::string topicName;
    topicName = "/" + model->GetName() + "/odometry";

    // Print loaded parameters
    gzdbg << "MovingTargetPlugin loaded for model [" << model->GetName()
          << "] with velocity: " << this->velocity
          << " and odometry topic: " << topicName << std::endl;

    // Initialize ROS node handle if not already initialized
    if (!ros::isInitialized())
    {
      ROS_FATAL_STREAM("ROS node not initialized. Make sure to call ros::init() before loading the plugin.");
      return;
    }

    this->rosNode.reset(new ros::NodeHandle(""));

    // Initialize publisher
    this->odomPub = this->rosNode->advertise<nav_msgs::Odometry>(topicName, 10);

    // Register update callback
    this->updateConnection = event::Events::ConnectWorldUpdateBegin(
        std::bind(&MovingTargetPlugin::OnUpdate, this));

    common::Time::MSleep(2000);  // Let world initialize
    this->lastUpdateTime = this->model->GetWorld()->SimTime();
  }

  void MovingTargetPlugin::OnUpdate()
  {
    common::Time currentTime = this->model->GetWorld()->SimTime();
    double timeDelta = (currentTime - this->lastUpdateTime).Double();

    if (timeDelta > 0)
    {
      // Update target position
      ignition::math::Pose3d currentPose = this->model->WorldPose();
      ignition::math::Vector3d newPosition = currentPose.Pos() + this->velocity * timeDelta;
      this->model->SetWorldPose(ignition::math::Pose3d(newPosition, currentPose.Rot()));
      this->lastUpdateTime = currentTime;

      // Publish odometry
      nav_msgs::Odometry odom;
      odom.header.stamp = ros::Time::now();
      odom.header.frame_id = "world";
      odom.child_frame_id = this->model->GetName();

      odom.pose.pose.position.x = newPosition.X();
      odom.pose.pose.position.y = newPosition.Y();
      odom.pose.pose.position.z = newPosition.Z();

      odom.pose.pose.orientation.x = currentPose.Rot().X();
      odom.pose.pose.orientation.y = currentPose.Rot().Y();
      odom.pose.pose.orientation.z = currentPose.Rot().Z();
      odom.pose.pose.orientation.w = currentPose.Rot().W();

      this->odomPub.publish(odom);
    }
  }

  GZ_REGISTER_MODEL_PLUGIN(MovingTargetPlugin);
}