#!/usr/bin/env python3
import math, random, rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from tf.transformations import quaternion_from_euler

def random_walk():
    rospy.init_node('random_random_walk_publisher')
    pub = rospy.Publisher('/cmd_path', Path, queue_size=1)
    rate = rospy.Rate(0.2)  # publish a new random path every 5 seconds

    # ────────────────
    # define your 2 m×2 m region here:
    x_min, x_max = 5.0, 9.0   # X between 5 and 7 meters
    y_min, y_max = -1.0, 1.0  # Y between –1 and +1 meters
    num_waypoints = 20        # how many waypoints per path
    # ────────────────

    while not rospy.is_shutdown():
        p = Path()
        p.header.stamp = rospy.Time.now()
        p.header.frame_id = "world"

        for _ in range(num_waypoints):
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            θ = random.uniform(-math.pi, math.pi)

            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = "world"
            pose.pose.position = Point(x, y, 0.0)
            q = quaternion_from_euler(0, 0, θ)
            pose.pose.orientation = Quaternion(*q)

            p.poses.append(pose)

        pub.publish(p)
        rospy.loginfo(f"[random_walk] published {num_waypoints} waypoints in box "
                      f"[{x_min},{x_max}]×[{y_min},{y_max}]")
        rate.sleep()

if __name__ == "__main__":
    try:
        random_walk()
    except rospy.ROSInterruptException:
        pass
