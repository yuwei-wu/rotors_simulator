#ifndef MOVING_TARGET_PLUGIN_H
#define MOVING_TARGET_PLUGIN_H

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/Plugin.hh>
#include <ignition/math/Vector3.hh>
#include <gazebo/common/Time.hh>

namespace gazebo
{
  class MovingTargetPlugin : public ModelPlugin
  {
  public:
    /// Constructor
    MovingTargetPlugin() = default;

    /// Destructor
    virtual ~MovingTargetPlugin() = default;

    /// Load the plugin (called by Gazebo)
    /// @param model Pointer to the parent model
    /// @param sdf Pointer to the SDF element
    void Load(physics::ModelPtr model, sdf::ElementPtr sdf) override;

    /// Update function called at every simulation iteration
    void OnUpdate();

  private:
    /// Pointer to the model this plugin is attached to
    physics::ModelPtr model;

    /// Pointer to the world update event connection
    event::ConnectionPtr updateConnection;

    /// Target's constant velocity vector
    ignition::math::Vector3d velocity;

    /// Last update time
    common::Time lastUpdateTime;


    int update_cnt_ = 0;
    int update_rate_ = 10;
  };
}

// Macro to register the plugin with Gazebo
// GZ_REGISTER_MODEL_PLUGIN(gazebo::MovingTargetPlugin)

#endif // MOVING_TARGET_PLUGIN_H