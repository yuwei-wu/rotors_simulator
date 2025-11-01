# Ego-Planner Integration with RotorS Simulator

## Overview

This integration connects the Ego-Planner motion planner with the RotorS quadrotor simulator in Gazebo using the aligned depth sensor for obstacle detection.

## Architecture

```
Aligned Depth Sensor → Ego-Planner → Bridge Node → RotorS Controller → Gazebo
(Point Cloud/Depth)     (Planning)   (Message Conv)  (Control)        (Simulation)
```

### Components:

1. **Aligned Depth Sensor**: Provides depth images and point clouds
   - Topics: `/hummingbird/vi_sensor/camera_depth/depth/disparity`, `/hummingbird/vi_sensor/camera_depth/depth/points`
   
2. **Ego-Planner**: Generates collision-free trajectories
   - Publishes: `/planning/pos_cmd` (quadrotor_msgs/PositionCommand)
   
3. **Bridge Node**: `ego_planner_bridge_node`
   - Converts ego-planner commands to rotors format
   - Subscribes: `/planning/pos_cmd`
   - Publishes: `/hummingbird/command/trajectory`
   
4. **Lee Position Controller**: Controls the quadrotor
   - Subscribes: `/hummingbird/command/trajectory`
   - Publishes: motor commands to Gazebo

## Configuration

### Platform Details:
- **Drone**: Hummingbird quadrotor with aligned depth sensor
- **World**: `outdoor_with_random_actor` (includes dynamic obstacles)
- **Initial Position**: x=-4.0, y=-4.0, z=0.8
- **Odometry Source**: Odometry sensor (`/hummingbird/odometry_sensor1/odometry`)
- **Depth Sensor Resolution**: 640x480
- **Camera Intrinsics**: fx=fy=462.14, cx=320.5, cy=240.5

## Usage

### 1. Build the workspace:
```bash
cd /home/bill/gz_ws
catkin build
source devel/setup.bash
```

### 2. Launch the integrated system:
```bash
roslaunch rotors_gazebo rotors_with_ego_planner.launch
```

### 3. Set a goal and fly:
- The simulation starts unpaused automatically
- In RViz, use "2D Nav Goal" tool to set a target position
- The drone will plan around obstacles (including moving actors) and execute the trajectory

### 4. Optional arguments:
```bash
# Launch without GUI (faster)
roslaunch rotors_gazebo rotors_with_ego_planner.launch gui:=false

# Change initial position
roslaunch rotors_gazebo rotors_with_ego_planner.launch init_x:=0.0 init_y:=0.0 init_z:=1.0

# Use different world
roslaunch rotors_gazebo rotors_with_ego_planner.launch world_name:=basic
```

## Troubleshooting

### Drone not responding to goals:
1. Check that all nodes are running: `rosnode list`
2. Verify topics: `rostopic list | grep -E "(planning|command)"`
3. Check bridge is converting messages: `rostopic echo /firefly/command/trajectory`

### Planning fails:
1. Ensure map is loaded: `rostopic echo /map_generator/global_cloud`
2. Check odometry: `rostopic echo /firefly/ground_truth/odometry`
3. Verify goal is within map bounds

### Build errors:
Make sure all dependencies are built:
```bash
catkin build quadrotor_msgs mav_msgs pose_utils
catkin build rotors_control
```

## Files Created/Modified

**New files:**
- `rotors_control/src/nodes/ego_planner_bridge_node.cpp` - Message converter
- `rotors_gazebo/launch/rotors_with_ego_planner.launch` - Integrated launch file

**Modified files:**
- `rotors_control/CMakeLists.txt` - Added bridge node
- `rotors_control/package.xml` - Added quadrotor_msgs dependency

## Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/planning/pos_cmd` | quadrotor_msgs/PositionCommand | Ego-planner output |
| `/firefly/command/trajectory` | trajectory_msgs/MultiDOFJointTrajectory | Controller input |
| `/firefly/ground_truth/odometry` | nav_msgs/Odometry | Drone state |
| `/move_base_simple/goal` | geometry_msgs/PoseStamped | Goal from RViz |

## Next Steps

1. Add depth camera or LIDAR for sensing
2. Tune planner parameters for your environment
3. Implement more complex missions with waypoints
4. Add visualization in RViz
