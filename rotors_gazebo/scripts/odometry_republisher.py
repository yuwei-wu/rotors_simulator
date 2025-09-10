#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry

class OdometryRepublisher:
    def __init__(self, car_name, rate_hz=30.0):
        self.car_name = car_name
        self.latest_msg = None

        # Subscriber (high-frequency source)
        rospy.Subscriber(f"/{car_name}/odometry", Odometry, self.callback)

        # Publisher (fixed-rate output)
        self.pub = rospy.Publisher(f"/{car_name}/odometry_throttled", Odometry, queue_size=10)

        # Timer to republish at exactly rate_hz
        rospy.Timer(rospy.Duration(1.0 / rate_hz), self.timer_callback)

    def callback(self, msg):
        """Cache the latest odometry message."""
        self.latest_msg = msg

    def timer_callback(self, event):
        """Republish the latest odometry message at fixed rate."""
        if self.latest_msg is not None:
            self.pub.publish(self.latest_msg)

if __name__ == "__main__":
    rospy.init_node("odometry_republisher")

    # Get params (so you can reuse for multiple cars)
    car_name = rospy.get_param("~car_name", "car_1")
    rate_hz = rospy.get_param("~rate_hz", 30.0)

    republisher = OdometryRepublisher(car_name, rate_hz)
    rospy.spin()
