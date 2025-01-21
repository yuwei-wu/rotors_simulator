#!/usr/bin/env python3

import rospy
import csv
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion

# File for logging
log_file_odom = "./drone_log.csv"
log_file_ground_truth = "./drone_ground_truth.csv"
log_file_target = "./target_log.csv"

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

# Callback for /odom topic
def odom_callback(msg, log_filename):
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
        
def car_position_callback(msg, log_filename):
    # The car's position is in the Odometry message
    # If using `gazebo/model_states`, you can also extract the position similarly
    car_index = msg.name.index("car_199")  # Replace "car_199" with the actual model name
    car_position = msg.pose[car_index].position

    # Create a Point message with the car's position
    x, y, z = car_position.x, car_position.y, car_position.z

    # Log data
    with open(log_filename, "a") as file:
        writer = csv.writer(file)
        writer.writerow([rospy.get_time(), x, y, z])

def main():
    rospy.init_node("my_logger", anonymous=True)
    init_csv_odom_file(log_file_odom)
    init_csv_odom_file(log_file_ground_truth)
    init_csv_position_file(log_file_target)

    # Subscribe to the odometry topic (change as needed)
    rospy.Subscriber("/hummingbird/odometry_sensor1/odometry", Odometry, lambda msg: odom_callback(msg, log_file_odom))
    rospy.Subscriber("/hummingbird/ground_truth/odometry", Odometry, lambda msg: odom_callback(msg, log_file_ground_truth))
    rospy.Subscriber('/gazebo/model_states', ModelStates, lambda msg: car_position_callback(msg, log_file_target))

    # Keep node running
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
