#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32MultiArray

class TrainingManager():
    def __init__(self):
        """
        Background thread to publish PyTorch model weights periodically.

        Args:
            topic_name (str): ROS topic name.
        """
        super(TrainingManager, self).__init__()
        self.train_running = False  # to denote whether the training on the other end is finished
        self.train_pub = rospy.Publisher("/training_start", Float32MultiArray, queue_size=10)

    
    def publish_training_start(self):
        # Publish a message on the training_start topic to fire centralized or fl training
        msg = Float32MultiArray()
        msg.data = [True]
        self.train_pub.publish(msg)
        self.train_running = True
        
    
    def check_train_running(self):
        return self.train_running


    def reset_train_running(self):
        self.train_running = False

