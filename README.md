------------------------
RotorS Instructions for Resilient Planner
===============

This is a customized version to run [VID-Fusion](https://github.com/ZJU-FAST-Lab/VID-Fusion) and [External Forces Resilient Planner](https://github.com/ZJU-FAST-Lab/forces_resilient_planner).


## 1. Installation Instructions

This simulation has been tested on Ubuntu 20.04 with ROS Noetic (and Ubuntu 18.04 with ROS Melodic). 
You can follow the instructions below to install and initialize ROS desktop full, additional ROS packages, catkin-tools, and wstool:

### Ubuntu 20.04 with ROS Noetic

```console
$ sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
$ sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
$ sudo apt update
$ sudo apt install ros-noetic-desktop-full ros-noetic-joy ros-noetic-octomap-ros ros-noetic-mavlink ros-noetic-mavros
$ sudo apt install python3-wstool python3-catkin-tools protobuf-compiler libgoogle-glog-dev ros-noetic-control-toolbox
$ sudo pip3 install python-uinput pygame
$ sudo rosdep init
$ rosdep update
$ echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
$ source ~/.bashrc
$ sudo apt install python3-rosinstall python3-rosinstall-generator build-essential
```

### Potential issues

#### 1. Issues: https://github.com/gazebosim/gz-sim/issues/2364
Solution:
```
sudo add-apt-repository ppa:kisak/kisak-mesa
sudo apt update
sudo apt upgrade
```

#### 2. Hil interface issue: https://aur.archlinux.org/packages/ros-noetic-rotors-hil-interface

replace with: 
```
git clone https://aur.archlinux.org/ros-noetic-rotors-hil-interface.git
```

### Ubuntu 18.04 with ROS Melodic

```console
$ sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
$ sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
$ sudo apt update
$ sudo apt install ros-melodic-desktop-full ros-melodic-joy ros-melodic-octomap-ros ros-melodic-mavlink ros-melodic-mavros
$ sudo apt install python-wstool python-catkin-tools protobuf-compiler libgoogle-glog-dev ros-melodic-control-toolbox
$ sudo pip install python-uinput pygame
$ sudo rosdep init
$ rosdep update
$ echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
$ source ~/.bashrc
$ sudo apt install python-rosinstall python-rosinstall-generator build-essential
```

2. If you don't have a ROS workspace yet, you can do so by: (It's better to directly create a new workspace.)

```console
$ mkdir -p ~/catkin_ws/src && cd ~/catkin_ws/
$ catkin init --workspace .
$ cd src && git clone https://github.com/yuwei-wu/rotors_simulator
$ cd rotors_simular
$ wstool init
$ wstool merge rotors_hil.rosinstall
$ wstool update
```

3. Build your workspace

```console
$ cd ~/catkin_ws/
$ catkin build
```

## 2. Run the costumed environment

We use the drone "hummingbird" with a vi sensor for testing. After following the instructions of RotorS setting, you can directly run: 

```
source devel/setup.bash && roslaunch rotors_gazebo fast_with_vi_sensor.launch
```

It will also launch a virtual joystick window, and you can press the keyboard to control the drone. 
(The joy interface is revised as position control rather than altitude control, which is easier for the interface.)

The important topics are:

For planner:
- subscribers:
  - `ground truth odom`:  /hummingbird/ground_truth/odometry
  - `depth camera`:  /hummingbird/vi_sensor/camera_depth/depth/disparity

- publishers:
  - `traj cmd`:  /hummingbird/command/trajectory

For VIO:
- subscribers:
  - `imu_topic`:  /hummingbird/ground_truth/imu
  - `image_topic`:  /hummingbird/vi_sensor/left/image_raw
  - `control_topic`:  /hummingbird/thrust

To get camera parameters:

`Camera info`: /hummingbird/vi_sensor/camera_depth/camera/camera_info


## 3. Set Your Camera Parameters

It's flexible to change its focal length and the position on the drone.

- internal parameters

In `rotors_description/urdf/component_snippets.xacro`, start with line 626, find a macro `vi_sensor_depth_macro` and revise its camera_plugin to change focal length and distortion.

Tutorials for more instructions: http://gazebosim.org/tutorials/?tut=ros_depth_camera


- external parameters

In `rotors_description/urdf/mav_with_vi_sensor.gazebo`, to revise the origin xyz and rpy.

```xml
  <!-- Mount a VI-sensor in front of the Firefly. -->
  <xacro:vi_sensor_macro
    namespace="${namespace}/vi_sensor"
    parent_link="${namespace}/base_link"
    enable_cameras="true"
    enable_depth="true"
    enable_ground_truth="true">
    <origin xyz="0.1 0.0 0.086" rpy="0.0 0.0 0.0" />
  </xacro:vi_sensor_macro>

```

## 4. Adjust External Forces

RotorS has two interfaces to add custom wind: the whole wind distribution or wind during time. For convenience, we also add two easier interfaces. 
You can both use `joystick` or `disturbance manager` to add external forces.

- By using joystick/keyboard: you can press y/u h/j n/m to add an external force in three dimension

- By disturbance manager: follow the example in disturbance_manager.cpp to add costumed force.


## 5. Other Notes

#### Customize the world file

VID-fusion need features to track, so it's better to change the texture in you environment. In `fast_quad.world`, I add several brick box with different texture, and their model can be found in `rotors_gazebo/models`


### 5.1 Drone's Mass

Because we use hummingbird for testing, when vi-sensor is placed in the CoG of the quadrotor, the rotors will appear in the camera image.
However if set in front, it will not stable because of the mass of vi-sensor and need to adjust the parameters of position controller.

To easily avoid the issue, we decrease the mass of vi-sensor near zero and calibrate the mass of drone. By testing during drone's hovering, the mass is mass: 0.745319 now. If you directly clone the original verision, you need to revise it in 


### 5.2 Drone's Color

In `rotors_description/urdf/hummingbird.xacro` to change the color of the rotors and body of quadrotor.

```xml
  <!-- Instantiate multirotor_base_macro once -->
  <xacro:multirotor_base_macro
    robot_namespace="${namespace}"
    mass="${mass}"
    body_width="${body_width}"
    body_height="${body_height}"
    use_mesh_file="${use_mesh_file}"
    color="Grey"
    mesh_file="${mesh_file}">
    <xacro:insert_block name="body_inertia" />
  </xacro:multirotor_base_macro>
```

### More gazebo models


- https://github.com/engcang/gazebo_maps

- https://github.com/mlherd/Dataset-of-Gazebo-Worlds-Models-and-Maps

- https://github.com/osrf/gazebo_models

- https://github.com/PX4/PX4-SITL_gazebo-classic/blob/main/worlds/

- https://github.com/KumarRobotics/mrsl_quadrotor



## About RotorS

RotorS is a MAV gazebo simulator.
It provides some multirotor models such as the [AscTec Hummingbird](http://www.asctec.de/en/uav-uas-drone-products/asctec-hummingbird/), the [AscTec Pelican](http://www.asctec.de/en/uav-uas-drone-products/asctec-pelican/), or the [AscTec Firefly](http://www.asctec.de/en/uav-uas-drone-products/asctec-firefly/), but the simulator is not limited for the use with these multicopters.

There are simulated sensors coming with the simulator such as an IMU, a generic odometry sensor, and the [VI-Sensor](http://wiki.ros.org/vi_sensor), which can be mounted on the multirotor.

This package also contains some example controllers, basic worlds, a joystick interface, and example launch files.

Below we provide the instructions necessary for getting started. See RotorS' wiki for more instructions and examples (https://github.com/ethz-asl/rotors_simulator/wiki).

If you are using this simulator within the research for your publication, please cite:
```bibtex
@Inbook{Furrer2016,
author="Furrer, Fadri
and Burri, Michael
and Achtelik, Markus
and Siegwart, Roland",
editor="Koubaa, Anis",
chapter="RotorS---A Modular Gazebo MAV Simulator Framework",
title="Robot Operating System (ROS): The Complete Reference (Volume 1)",
year="2016",
publisher="Springer International Publishing",
address="Cham",
pages="595--625",
isbn="978-3-319-26054-9",
doi="10.1007/978-3-319-26054-9_23",
url="http://dx.doi.org/10.1007/978-3-319-26054-9_23"
}
```
