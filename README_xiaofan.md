# Target tracking data collection

### Update

Just run one launch to run both the sim and the data logger together:
```
roslaunch rotors_gazebo mav_swarm_all.launch
```


### Features

* Targets do **not move autonomously**. They will only start moving after you set a **2D Nav Goal** in RViz (published to `/move_base_simple/goal`).
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
To run the YOLO detection, you need to download the `best_yolo.pt` from [here](https://drive.google.com/file/d/13hKl5SC1ntilpZSolp6-cfXE_w6vSSQH/view?usp=sharing) and the `best_model.pth` from [here](https://drive.google.com/file/d/1BcnYvoRmn2784whT3mFO7aKgoJ9nPx3D/view?usp=sharing). Put both models under `rotors_gazebo/scripts/`.

Run the following command under `rotors_gazebo/scripts/`:
```bash
python3 data_logger.py
```
Line 148 of this script will give the predicted trajectory.

Open two other terminals to run the following:
```bash
# One terminal running ros simulation
roslaunch rotors_gazebo mav_swarm.launch
# One terminal running Yuwei's planner
python3 tracker_server.py
```

### File Locations

* File for logging script: https://github.com/yuwei-wu/rotors_simulator/blob/xiaofan/rotors_gazebo/scripts/my_logger.py
* File for the camera setup: https://github.com/yuwei-wu/rotors_simulator/blob/xiaofan/rotors_description/urdf/mav_with_camera.gazebo

* File for the agricultural world: https://github.com/yuwei-wu/rotors_simulator/blob/xiaofan/rotors_gazebo/worlds/agriculture.world
* File for the target: https://github.com/yuwei-wu/rotors_simulator/blob/xiaofan/rotors_gazebo/models/car_199/model.sdf
* File for the moving target plugin: [.h file](https://github.com/yuwei-wu/rotors_simulator/blob/xiaofan/rotors_gazebo_plugins/include/rotors_gazebo_plugins/moving_target_plugin.h), [.cpp file](https://github.com/yuwei-wu/rotors_simulator/blob/xiaofan/rotors_gazebo_plugins/src/moving_target_plugin.cpp)

### Other Important Settings
* Target speed is set in https://github.com/yuwei-wu/rotors_simulator/blob/xiaofan/rotors_gazebo/models/car_199/model.sdf
* Target initial location is set in https://github.com/yuwei-wu/rotors_simulator/blob/xiaofan/rotors_gazebo/worlds/agriculture.world


### Runing multi-threaded version
* run this instead of data_logger.py
```bash
#Separate terminal
roslaunch rotors_gazebo multi_drone.launch\
```

* File for multi_drone.launch: https://github.com/yuwei-wu/rotors_simulator/blob/xiaofan/rotors_gazebo/launch/multi_drone.launch
* File for drone_bbox_node.py: https://github.com/yuwei-wu/rotors_simulator/blob/xiaofan/rotors_gazebo/scripts/drone_bbox_node.py
