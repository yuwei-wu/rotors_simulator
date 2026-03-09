#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry

class OdometryRepublisher:
    def __init__(self, input_topic, output_topic, rate_hz=30.0):
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.latest_msg = None

        # Subscriber (high-frequency source)
        rospy.Subscriber(input_topic, Odometry, self.callback)

        # Publisher (fixed-rate output)
        self.pub = rospy.Publisher(output_topic, Odometry, queue_size=10)

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
    input_topic = rospy.get_param("~input_topic", f"/{car_name}/odometry")
    output_topic = rospy.get_param("~output_topic", f"/{car_name}/odometry_throttled")
    rate_hz = rospy.get_param("~rate_hz", 30.0)

    republisher = OdometryRepublisher(input_topic, output_topic, rate_hz)
    rospy.spin()
