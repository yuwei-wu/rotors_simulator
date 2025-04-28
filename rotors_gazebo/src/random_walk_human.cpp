#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <random>

namespace gazebo
{
  class RandomWalkHuman : public ModelPlugin
  {
    public: void Load(physics::ModelPtr _model, sdf::ElementPtr /*_sdf*/)
    {
      this->model = _model;
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          std::bind(&RandomWalkHuman::OnUpdate, this));
      lastUpdateTime = _model->GetWorld()->SimTime();
    }

    public: void OnUpdate()
    {
      common::Time currentTime = this->model->GetWorld()->SimTime();
      if ((currentTime - lastUpdateTime).Double() > 2.0)
      {
        double angle = distribution(generator) * 2 * M_PI;
        double distance = 0.5 + distribution(generator) * 0.5; // 0.5 to 1.0 meters

        ignition::math::Pose3d pose = this->model->WorldPose();
        pose.Pos().X() += distance * cos(angle);
        pose.Pos().Y() += distance * sin(angle);

        this->model->SetWorldPose(pose);

        lastUpdateTime = currentTime;
      }
    }

    private: physics::ModelPtr model;
    private: event::ConnectionPtr updateConnection;
    private: common::Time lastUpdateTime;

    private: std::default_random_engine generator;
    private: std::uniform_real_distribution<double> distribution{0.0, 1.0};
  };

  GZ_REGISTER_MODEL_PLUGIN(RandomWalkHuman)
}
