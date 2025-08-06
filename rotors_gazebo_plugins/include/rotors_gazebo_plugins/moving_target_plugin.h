#ifndef MOVING_TARGET_PLUGIN_H
#define MOVING_TARGET_PLUGIN_H

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/Plugin.hh>
#include <ignition/math/Vector3.hh>
#include <gazebo/common/Time.hh>

#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>

namespace gazebo
{
  class MovingTargetPlugin : public ModelPlugin
  {
  public:
    /// Constructor
    MovingTargetPlugin() = default;

    /// Destructor
    virtual ~MovingTargetPlugin() = default;

    /// Called when the plugin is loaded
    /// @param model Pointer to the parent model
    /// @param sdf Pointer to the SDF element
    void Load(physics::ModelPtr model, sdf::ElementPtr sdf) override;

    /// Called at every simulation iteration
    void OnUpdate();

  private:
    /// Callback when a goal message is received (acts as trigger)
    /// @param msg PoseStamped message (only acts as a trigger, position ignored)
    void OnGoalReceived(const geometry_msgs::PoseStamped::ConstPtr& msg);

    /// Publishes the model's odometry
    void PublishOdometry(const ignition::math::Pose3d& pose);

    /// Pointer to the model this plugin is attached to
    physics::ModelPtr model;

    /// Pointer to the world update event connection
    event::ConnectionPtr updateConnection;

    /// Target's constant velocity vector (loaded from SDF)
    ignition::math::Vector3d velocity;

    /// Last update time for motion integration
    common::Time lastUpdateTime;

    /// Flag indicating whether the target should move
    bool isMoving = false;

    /// ROS Node Handle
    std::unique_ptr<ros::NodeHandle> rosNode;

    /// ROS Subscriber to trigger movement
    ros::Subscriber goalSub;

    /// ROS Publisher for odometry
    ros::Publisher odomPub;
  };
}

// Macro to register the plugin with Gazebo
// GZ_REGISTER_MODEL_PLUGIN(gazebo::MovingTargetPlugin)

#endif // MOVING_TARGET_PLUGIN_H