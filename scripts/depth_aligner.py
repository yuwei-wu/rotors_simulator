#!/usr/bin/env python3

import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

class DepthImageAligner:
    def __init__(self):
        rospy.init_node('depth_image_aligner')
        
        self.bridge = CvBridge()
        
        # Subscribers for RGB and depth images
        self.rgb_sub = message_filters.Subscriber('/hummingbird/rgb_camera/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/hummingbird/vi_sensor/camera_depth/depth/disparity', Image)
        
        # Synchronize RGB and depth images
        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], 10, 0.1)
        self.ts.registerCallback(self.callback)
        
        # Publishers for aligned images
        self.aligned_depth_pub = rospy.Publisher('/hummingbird/aligned_depth/image_raw', Image, queue_size=1)
        self.aligned_rgb_pub = rospy.Publisher('/hummingbird/aligned_rgb/image_raw', Image, queue_size=1)
        
        rospy.loginfo("Depth Image Aligner started")
    
    def callback(self, rgb_msg, depth_msg):
        try:
            # Convert ROS images to OpenCV format
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
            
            # Get image dimensions
            rgb_height, rgb_width = rgb_image.shape[:2]
            depth_height, depth_width = depth_image.shape[:2]
            
            # Resize depth image to match RGB image size
            aligned_depth = cv2.resize(depth_image, (rgb_width, rgb_height), interpolation=cv2.INTER_NEAREST)
            
            # Convert back to ROS messages
            aligned_depth_msg = self.bridge.cv2_to_imgmsg(aligned_depth, "32FC1")
            aligned_depth_msg.header = rgb_msg.header
            
            aligned_rgb_msg = rgb_msg  # RGB is already in the right frame
            
            # Publish aligned images
            self.aligned_depth_pub.publish(aligned_depth_msg)
            self.aligned_rgb_pub.publish(aligned_rgb_msg)
            
        except Exception as e:
            rospy.logerr(f"Error in depth alignment: {e}")

if __name__ == '__main__':
    try:
        aligner = DepthImageAligner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
