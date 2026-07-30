#!/usr/bin/env python3
"""Keeps the 'random_walker' actor wandering inside a bounded rectangular
region, by periodically appending a fresh batch of random waypoints to its
gazebo_ros_actor_command path queue (same append-only mechanism used by
road_walker_publisher.py -- see that file for why the republish has to
happen before the current batch is exhausted).

Same idea as gazebo-ros-actor-plugin/scripts/random_walk.py, but with the
region and topic exposed as ROS params instead of hardcoded.

Pass ~seed to get the same sequence of "random" waypoints across runs --
important for a fair llm_on vs llm_off comparison (see
flight_data_logger.py / analyze_flight_logs.py), so both conditions face
the exact same pedestrian motion and the only thing that differs is
whether the LLM bbox pipeline was running.
"""
import math
import random

import rospy
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from nav_msgs.msg import Path
from tf.transformations import quaternion_from_euler


def main():
    rospy.init_node("region_random_walker_publisher")

    topic = rospy.get_param("~topic", "/random_walker/cmd_path")
    x_min = rospy.get_param("~x_min", 7.0)
    x_max = rospy.get_param("~x_max", 17.0)
    y_min = rospy.get_param("~y_min", 3.0)
    y_max = rospy.get_param("~y_max", 9.0)
    num_waypoints = rospy.get_param("~num_waypoints", 8)
    interval = rospy.get_param("~interval", 8.0)
    seed = rospy.get_param("~seed", None)
    if seed is not None:
        random.seed(seed)

    pub = rospy.Publisher(topic, Path, queue_size=1)
    rospy.sleep(1.0)

    rospy.loginfo(f"[region_random_walker] region x[{x_min},{x_max}] "
                  f"y[{y_min},{y_max}], {num_waypoints} waypoints every "
                  f"{interval}s on {topic}")
    rate = rospy.Rate(1.0 / interval)
    while not rospy.is_shutdown():
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = "world"
        for _ in range(num_waypoints):
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            yaw = random.uniform(-math.pi, math.pi)
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = "world"
            pose.pose.position = Point(x, y, 0.0)
            pose.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, yaw))
            path.poses.append(pose)
        pub.publish(path)
        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
