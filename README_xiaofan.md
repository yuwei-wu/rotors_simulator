# Target tracking data collection

### Update

Just run one launch to run both the sim and the data logger together:
```
roslaunch rotors_gazebo mav_swarm_all.launch
```


### Features

* Targets are Gazebo car models driven by `moving_target_plugin`.
* Each car publishes live odometry on `/car_X/odometry`.
* When using the updated `drone-fl` planner, targets can start automatically without manually clicking a **2D Nav Goal** in RViz. The planner publishes the `/move_base_simple/goal` trigger itself by default.
* Added **predicted target trajectory visualization**:

  * The prediction is visualized as markers in RViz.
  * Note: The relation of X and Y coordinates in the prediction seems to be in a **local or relative frame**, not the global world frame. Ensure proper transformation if you need them in the global coordinate system.
 
  
### Getting Started

Prerequisite: Need to setup the virtual joy stick ([wiki](https://github.com/ethz-asl/rotors_simulator/wiki/Setup-virtual-keyboard-joystick))

After successful catkin build, you can run the simulator with the following command:

```bash
roslaunch rotors_gazebo mav_with_keyboard.launch mav_name:=hummingbird world_name:=agriculture
```

The logging script can be triggered in a new terminal window, with the following command:

```bash
# Suppose the current path is ~/catkin_ws
python3 src/rotors_simulator/rotors_gazebo/scripts/my_logger.py

```

### Integration
I am using Python 3.8 to run the detection model. The requirements can be found in `rotors_gazebo/scripts/requirements.txt`.
To run the YOLO detection, you need to download the `best_yolo.pt` from [here](https://drive.google.com/file/d/13hKl5SC1ntilpZSolp6-cfXE_w6vSSQH/view?usp=sharing) and the `best_model.pth` from [here](https://drive.google.com/file/d/1tC1POn3bKJMiN3hDJFuS1JfeggC3f_0n/view?usp=sharing). Put both models under `rotors_gazebo/pred_model_ckpt/`.

Open three terminals to run the following:
```bash
# One terminal running ros simulation
roslaunch rotors_gazebo mav_swarm.launch
# One terminal running Yuwei's planner
python3 tracker_server.py
# One terminal running multi-thred yolo detection and trajectory prediction
roslaunch rotors_gazebo multi_drone.launch
```

### Planner With Gazebo Cars

The `drone-fl` planner can now use the Gazebo cars directly as live targets.

What is connected:

* `moving_target_plugin` moves each car model and publishes `/car_X/odometry`
* `tracker_server.py` subscribes to `/car_X/odometry`
* In `ros simulation`, the planner now uses the live car odometry as target ground truth instead of propagating an internal synthetic target model
* The planner also auto-publishes `/move_base_simple/goal` once the ROS topics are ready, so the cars start moving automatically by default

Planner config requirements in `drone-fl/planner/config/<exp>.yaml`:

```yaml
exp: "ros simulation"
targetID: [1, 2, 3]
targetID_car: [1, 2, 3]
target_autostart: true
fusion_method: "gt"   # optional if you want to test using ground truth targets only
```

Topic mapping:

* `targetID: [1, 2, 3]` means the planner subscribes to:
  * `/car_1/odometry`
  * `/car_2/odometry`
  * `/car_3/odometry`

Minimal run sequence:

Terminal 1:
```bash
cd ~/Code/xiaofan_ws
source devel/setup.bash
roslaunch rotors_gazebo mav_swarm.launch
```

Terminal 2:
```bash
cd ~/Code/xiaofan_ws/drone-fl/planner/script
python3 tracker_server.py --exp_name exp1
```

Terminal 3, only if you want detector-based predicted trajectories:
```bash
cd ~/Code/xiaofan_ws
source devel/setup.bash
roslaunch rotors_gazebo multi_drone.launch
```

Notes:

* If `fusion_method: "gt"`, the planner can run from car odometry alone and does not need `multi_drone.launch`.
* If `fusion_method` is `average` or `kalman`, the planner waits for `/droneX/pred_traj` messages from the detector stack.
* If you want the old manual start behavior, set `target_autostart: false` and publish a `2D Nav Goal` in RViz yourself.
* The current planner assumes car topics are named `/car_<id>/odometry`.

### Running Complete Experiments (End-to-End FL + ROS)

Four experiments are available: `exp0` (2 drones, 2 targets), `exp1` (3 drones, 3 targets), `exp2` (4 drones, 6 targets), and `scarab`. Each requires **four terminals**.

#### Terminal 1 — Gazebo Simulation

```bash
cd ~/catkin_ws
source devel/setup.bash

# Pick one:
roslaunch rotors_gazebo mav_swarm_0.launch      # exp0
roslaunch rotors_gazebo mav_swarm_1.launch      # exp1
roslaunch rotors_gazebo mav_swarm_2.launch      # exp2
roslaunch rotors_gazebo mav_swarm_scarab.launch  # scarab
```

#### Terminal 2 — Drone Processors (Detection + Prediction + FL Weight Receiving)

```bash
cd ~/catkin_ws
source devel/setup.bash

# Pick the matching experiment. Set arguments as needed:
#   learning:  frozen (no training) | dronefl (online federated learning)
#   fl_method: fedavg | fedper | fedprox  (only matters when learning:=dronefl)
#   adain:     true | false

# Examples:
roslaunch rotors_gazebo multi_drone_0.launch learning:=dronefl fl_method:=fedper adain:=true
roslaunch rotors_gazebo multi_drone_1.launch learning:=dronefl fl_method:=fedper adain:=true
roslaunch rotors_gazebo multi_drone_2.launch learning:=dronefl fl_method:=fedper adain:=true
roslaunch rotors_gazebo multi_drone_scarab.launch learning:=dronefl fl_method:=fedavg
```

#### Terminal 3 — Planner (Tracker Server)

```bash
cd ~/catkin_ws/drone-fl/planner/script

# Pick the matching experiment:
python3 tracker_server.py --exp_name exp0
python3 tracker_server.py --exp_name exp1
python3 tracker_server.py --exp_name exp2
python3 tracker_server.py --exp_name scarab
```

#### Terminal 4 — Federated Learning Training (only when `learning:=dronefl`)

```bash
cd ~/catkin_ws/drone-fl

# Usage: bash scripts/run_fed_ros_<exp>.sh <method> [adain]
#   method: fedavg | fedper | fedprox
#   adain:  optional, pass "adain" to enable AdaIN

# Examples:
bash scripts/run_fed_ros_0.sh fedavg
bash scripts/run_fed_ros_0.sh fedper adain
bash scripts/run_fed_ros_1.sh fedper adain
bash scripts/run_fed_ros_2.sh fedper adain
bash scripts/run_fed_ros_2.sh fedavg
```

**Important:** Make sure the `fl_method` in Terminal 2 matches the `method` in Terminal 4. For FedPer, each drone subscribes to its own per-client weight topic (`/model_weights/client_<i>`), so the FL side must publish per-client weights.

---

### File Locations

* File for logging script: https://github.com/yuwei-wu/rotors_simulator/blob/xiaofan/rotors_gazebo/scripts/my_logger.py
* File for the camera setup: https://github.com/yuwei-wu/rotors_simulator/blob/xiaofan/rotors_description/urdf/mav_with_camera.gazebo

* File for the agricultural world: https://github.com/yuwei-wu/rotors_simulator/blob/xiaofan/rotors_gazebo/worlds/agriculture.world
* File for the target: https://github.com/yuwei-wu/rotors_simulator/blob/xiaofan/rotors_gazebo/models/car_199/model.sdf
* File for the moving target plugin: [.h file](https://github.com/yuwei-wu/rotors_simulator/blob/xiaofan/rotors_gazebo_plugins/include/rotors_gazebo_plugins/moving_target_plugin.h), [.cpp file](https://github.com/yuwei-wu/rotors_simulator/blob/xiaofan/rotors_gazebo_plugins/src/moving_target_plugin.cpp)

### Other Important Settings
* Target speed is set in https://github.com/yuwei-wu/rotors_simulator/blob/xiaofan/rotors_gazebo/models/car_199/model.sdf
* Target initial location is set in https://github.com/yuwei-wu/rotors_simulator/blob/xiaofan/rotors_gazebo/worlds/agriculture.world
* File for multi_drone.launch: https://github.com/yuwei-wu/rotors_simulator/blob/xiaofan/rotors_gazebo/launch/multi_drone.launch
* File for drone_bbox_node.py: https://github.com/yuwei-wu/rotors_simulator/blob/xiaofan/rotors_gazebo/scripts/drone_bbox_node.py
