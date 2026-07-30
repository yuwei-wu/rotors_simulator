#!/usr/bin/env python3
"""Keeps the 'road_walker' actor walking back and forth along a fixed line
(a "sidewalk"), instead of the bounded random walk used for 'random_walker'.

gazebo_ros_actor_command's path mode APPENDS incoming waypoints to its
internal target list rather than replacing it, and stops once the list is
exhausted -- so to get a perpetual back-and-forth, we periodically append
one more leg, timed to arrive well before the actor runs out of targets.
"""
import math

import rospy
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from nav_msgs.msg import Path
from tf.transformations import quaternion_from_euler


def make_pose(x, y, yaw):
    pose = PoseStamped()
    pose.header.stamp = rospy.Time.now()
    pose.header.frame_id = "world"
    pose.pose.position = Point(x, y, 0.0)
    pose.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, yaw))
    return pose


def main():
    rospy.init_node("road_walker_publisher")

    topic = rospy.get_param("~topic", "/road_walker/cmd_path")
    x0 = rospy.get_param("~x0", 2.0)
    y0 = rospy.get_param("~y0", -6.0)
    x1 = rospy.get_param("~x1", 22.0)
    y1 = rospy.get_param("~y1", -6.0)
    speed = rospy.get_param("~speed", 1.2)  # must match the actor's linear_velocity

    leg_length = math.hypot(x1 - x0, y1 - y0)
    leg_time = leg_length / max(speed, 1e-3)
    # Republish well before the current leg finishes, so the actor never runs
    # out of targets and stalls.
    interval = max(2.0, leg_time * 0.5)

    pub = rospy.Publisher(topic, Path, queue_size=1)
    rospy.sleep(1.0)  # let the subscriber connect before the first publish

    forward = True
    rospy.loginfo(f"[road_walker] {leg_length:.1f}m leg, ~{leg_time:.1f}s, "
                  f"republishing every {interval:.1f}s on {topic}")
    rate = rospy.Rate(1.0 / interval)
    while not rospy.is_shutdown():
        (sx, sy), (ex, ey) = ((x0, y0), (x1, y1)) if forward else ((x1, y1), (x0, y0))
        yaw = math.atan2(ey - sy, ex - sx)

        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = "world"
        path.poses.append(make_pose(ex, ey, yaw))
        pub.publish(path)

        forward = not forward
        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
