#include "rotors_gazebo_plugins/moving_target_plugin.h"

namespace gazebo
{
  // Load function: Called once when the plugin is loaded
  void MovingTargetPlugin::Load(physics::ModelPtr model, sdf::ElementPtr sdf)
  {
    // Store the model pointer
    this->model = model;

    // Initialize velocity to default values or values from SDF
    this->velocity = ignition::math::Vector3d(0, 0, 0); // Default velocity

    // Check for velocity settings in SDF
    if (sdf->HasElement("velocity"))
    {
      this->velocity = sdf->Get<ignition::math::Vector3d>("velocity");
    }

    // Print confirmation to console
    gzdbg << "MovingTargetPlugin loaded with velocity: " 
          << this->velocity << std::endl;

    // Connect to the world update event
    this->updateConnection = event::Events::ConnectWorldUpdateBegin(
        std::bind(&MovingTargetPlugin::OnUpdate, this));


    // sleep for 2 seconds to let the world initialize
    common::Time::MSleep(2000);

    this->lastUpdateTime = this->model->GetWorld()->SimTime();
  }

  // OnUpdate function: Called every simulation iteration
  void MovingTargetPlugin::OnUpdate()
  {
    // Get the current simulation time
    common::Time currentTime = this->model->GetWorld()->SimTime();

    // Calculate the time delta
    double timeDelta = (currentTime - this->lastUpdateTime).Double();

    // Update the target's position
    if (timeDelta > 0)
    {
      update_cnt_ += 1;


      // Get the current position
      ignition::math::Pose3d currentPose = this->model->WorldPose();


      if (update_cnt_ % update_rate_ != 0)
      {
        // Set the new position
        this->model->SetWorldPose(ignition::math::Pose3d(currentPose.Pos(), currentPose.Rot()));

        // Update the last update time
        this->lastUpdateTime = currentTime;
      }
      else
      {
        // Calculate the new position
        ignition::math::Vector3d newPosition = currentPose.Pos() + this->velocity * timeDelta;

        // Print confirmation to console
        //gzdbg << "Current position:" << currentPose << " New position: " << newPosition << std::endl;
        // print with timestamp
        gzdbg << "[" << currentTime << "] Current position:" << currentPose << " New position: " << newPosition << std::endl;

        // Set the new position
        this->model->SetWorldPose(ignition::math::Pose3d(newPosition, currentPose.Rot()));

        // Update the last update time
        this->lastUpdateTime = currentTime;
      }

    }
  }

// Register the plugin with Gazebo
GZ_REGISTER_MODEL_PLUGIN(MovingTargetPlugin);
}