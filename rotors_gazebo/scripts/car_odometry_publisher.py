#!/usr/bin/env python3
import rospy
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry


class CarOdometryPublisher:
    def __init__(self, car_name, rate_hz):
        self.car_name = car_name
        self.latest_odom = None
        self.model_index = None

        self.odom_pub = rospy.Publisher(f"/{car_name}/odometry", Odometry, queue_size=10)
        self.odom_throttled_pub = rospy.Publisher(
            f"/{car_name}/odometry_throttled",
            Odometry,
            queue_size=10,
        )

        rospy.Subscriber("/gazebo/model_states", ModelStates, self._model_states_callback, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / max(rate_hz, 1e-3)), self._timer_callback)

    def _model_states_callback(self, msg):
        if self.model_index is None or self.model_index >= len(msg.name) or msg.name[self.model_index] != self.car_name:
            try:
                self.model_index = msg.name.index(self.car_name)
            except ValueError:
                self.model_index = None
                return

        odom_msg = Odometry()
        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.header.frame_id = "world"
        odom_msg.child_frame_id = self.car_name
        odom_msg.pose.pose = msg.pose[self.model_index]
        odom_msg.twist.twist = msg.twist[self.model_index]
        self.latest_odom = odom_msg

    def _timer_callback(self, _event):
        if self.latest_odom is None:
            return
        self.odom_pub.publish(self.latest_odom)
        self.odom_throttled_pub.publish(self.latest_odom)


if __name__ == "__main__":
    rospy.init_node("car_odometry_publisher")
    car_name = rospy.get_param("~car_name", "car_1")
    rate_hz = float(rospy.get_param("~rate_hz", 30.0))
    CarOdometryPublisher(car_name, rate_hz)
    rospy.spin()
