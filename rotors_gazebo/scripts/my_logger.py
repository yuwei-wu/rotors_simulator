#!/usr/bin/env python3

import rospy
import csv
import cv2
import os
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Header
from tf.transformations import euler_from_quaternion
from cv_bridge import CvBridge
import message_filters

# File for logging
log_file_odom = "./drone_log.csv"
log_file_ground_truth = "./drone_ground_truth.csv"
log_file_target = "./target_log.csv"
log_image_dir = "./drone_images"


def init_csv_odom_file(filename):
    # Initialize CSV file
    with open(filename, "w") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "X", "Y", "Z", "Roll", "Pitch", "Yaw",
                         "Linear_Vel_X", "Linear_Vel_Y", "Linear_Vel_Z",
                         "Angular_Vel_X", "Angular_Vel_Y", "Angular_Vel_Z"])
        
def init_csv_position_file(filename):
    # Initialize CSV file
    with open(filename, "w") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "X", "Y", "Z"])

# Callback for synchronized messages
def synchronized_callback(odom_msg, ground_truth_msg, model_states_msg, image_msg):

    timestamp = rospy.get_time()  # Use a single timestamp for all logs
    # Add a fake timestamp if missing
    if not hasattr(model_states_msg, "header"):
        model_states_msg.header = Header()
        model_states_msg.header.stamp = rospy.Time.now()
    rospy.loginfo('sychronization called at ', timestamp)

    # --- Process Odometry Data ---
    def process_odom(msg, log_filename):
        # Extract position
        position = msg.pose.pose.position
        x, y, z = position.x, position.y, position.z

        # Extract orientation (quaternion to Euler angles)
        orientation = msg.pose.pose.orientation
        quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
        roll, pitch, yaw = euler_from_quaternion(quaternion)

        # Extract velocity
        velocity = msg.twist.twist
        linear_velocity = velocity.linear
        angular_velocity = velocity.angular

        # Log data
        with open(log_filename, "a") as file:
            writer = csv.writer(file)
            writer.writerow([rospy.get_time(), x, y, z, roll, pitch, yaw,
                            linear_velocity.x, linear_velocity.y, linear_velocity.z,
                            angular_velocity.x, angular_velocity.y, angular_velocity.z])
    
    process_odom(odom_msg, log_file_odom)
    process_odom(ground_truth_msg, log_file_ground_truth)

    # --- Process Car Position ---
    # The car's position is in the Odometry message
    # If using `gazebo/model_states`, you can also extract the position similarly
    try:
        car_index = model_states_msg.name.index("car_199")  # Replace with actual model name if different
        car_position = model_states_msg.pose[car_index].position

        # Create a Point message with the car's position
        x, y, z = car_position.x, car_position.y, car_position.z

        # Log data
        with open(log_file_target, "a") as file:
            writer = csv.writer(file)
            writer.writerow([rospy.get_time(), x, y, z])
    except ValueError:
        rospy.logwarn("Car model not found in /gazebo/model_states")

    # --- Process Image Data ---
    bridge = CvBridge()

    try:
        # Convert the ROS image message to an OpenCV image
        cv_image = bridge.imgmsg_to_cv2(image_msg, "bgr8")

        # Get the current timestamp for unique filenames
        filename = os.path.join(log_image_dir, "{:.6f}.png".format(rospy.get_time()))

        # Save the image
        cv2.imwrite(filename, cv_image)

        rospy.loginfo("Saved image to {}".format(filename))

    except Exception as e:
        rospy.logerr("Failed to convert image: {}".format(e))


def main():
    rospy.init_node("my_logger", anonymous=True)

    # Synchronized rate
    sync_rate = rospy.Rate(30)  # 30Hz

    # Initialize CSV logs
    init_csv_odom_file(log_file_odom)
    init_csv_odom_file(log_file_ground_truth)
    init_csv_position_file(log_file_target)

    # Use message_filters to subscribe to multiple topics
    odom_sub = message_filters.Subscriber("/hummingbird/odometry_sensor1/odometry", Odometry)
    ground_truth_sub = message_filters.Subscriber("/hummingbird/ground_truth/odometry", Odometry)
    model_states_sub = message_filters.Subscriber("/gazebo/model_states", ModelStates)
    image_sub = message_filters.Subscriber("/hummingbird/camera_nadir/image_raw", Image)

    # Synchronize messages
    sync = message_filters.TimeSynchronizer([odom_sub, ground_truth_sub, model_states_sub, image_sub], queue_size=100)
    sync.registerCallback(synchronized_callback)

    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
