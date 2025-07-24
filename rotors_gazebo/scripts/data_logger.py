#!/usr/bin/env python3

import rospy
import csv
import cv2
import os
import numpy as np
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from tf.transformations import euler_from_quaternion
from cv_bridge import CvBridge
import message_filters
import threading
import torch
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image as PILImage

# File for logging
log_file_odom = "./drone_log.csv"
log_file_ground_truth = "./drone_ground_truth.csv"
log_file_target_1 = "./target_1_log.csv"
log_file_target_2 = "./target_2_log.csv"
log_file_target_3 = "./target_3_log.csv"
log_image_dir = "./drone_images"
last_time = 0

transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize to model's input
    transforms.ToTensor(),  # Convert to tensor (already done in dataset)
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_num = 3  # Number of targets to detect
yolo_model = None  # Placeholder for YOLO model
yolo_model_name = 'best_yolo.pt'

if not os.path.exists(log_image_dir):
    os.makedirs(log_image_dir)

def init_csv_odom_file(filename):
    # Initialize CSV file
    with open(filename, "w") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "X", "Y", "Z", "Roll", "Pitch", "Yaw",
                         "Linear_Vel_X", "Linear_Vel_Y", "Linear_Vel_Z",
                         "Angular_Vel_X", "Angular_Vel_Y", "Angular_Vel_Z"])

# Callback for synchronized messages
def synchronized_callback(odom_msg, ground_truth_msg, car_msg1, car_msg2, car_msg3, image_msg):

    #rate = rospy.Rate(10) # 10hz

    global last_time # Use global variable to keep track of time

    timestamp = rospy.get_time()  # Use a single timestamp for all logs
    rospy.loginfo("sychronization called at {}, the time duration is {} ms".format(timestamp, 
                                                                                   (timestamp - last_time) * 1000))
    last_time = timestamp

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
            writer.writerow([timestamp, x, y, z, roll, pitch, yaw,
                            linear_velocity.x, linear_velocity.y, linear_velocity.z,
                            angular_velocity.x, angular_velocity.y, angular_velocity.z])
    
    process_odom(odom_msg, log_file_odom)
    process_odom(ground_truth_msg, log_file_ground_truth)
    process_odom(car_msg1, log_file_target_1)
    process_odom(car_msg2, log_file_target_2)
    process_odom(car_msg3, log_file_target_3)
    print('process odom done')

    # --- Process Image Data ---
    bridge = CvBridge()

    try:
        # Convert the ROS image message to an OpenCV image
        cv_image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
        print('cv image ready', cv_image.shape)
        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        print('image rgb ready', image_rgb.shape)
        pil_image = PILImage.fromarray(image_rgb)

        image_tensor = transform(pil_image)

        # Step 4: Add batch dimension and move to device
        image_tensor = image_tensor.unsqueeze(0).to(device)  # Shape: (1, 3, 640, 640)

        # Step 5: Perform YOLO detection
        img_xywhn, detect_masks = yolo_detect(yolo_model, image_tensor, target_num, device)
        print(img_xywhn)

        # Get the current timestamp for unique filenames
        #filename = os.path.join(log_image_dir, "{:.6f}.png".format(timestamp))

        # Save the image
        #cv2.imwrite(filename, cv_image)

        #rospy.loginfo("Saved image to {}".format(filename))

    except Exception as e:
        rospy.logerr("Failed to convert image: {}".format(e))


    #rate.sleep()  # Sleep to maintain the rate

def yolo_detect(yolo_model, x_images, n_target, device):

    # Image input shape to YOLO: (bsz * win_size, 3, 640, 640)
    img_results = yolo_model.predict(source=x_images.view(-1, 3, 640, 640), verbose=False)

    # Find the detection results
    img_xywhn, detect_masks = [], []
    for result in img_results:
        detections, detect_mask = sort_detections(result, n_target, device)  
        # Output shape for detections: [n_target, 4]
        # Output shape for detect_mask: [n_target]
        
        img_xywhn.append(detections) 
        detect_masks.append(detect_mask)
    
    return torch.stack(img_xywhn), torch.stack(detect_masks)  # Return shape: [bsz * win_size, n_target, 4]


def sort_detections(result, n_target, device):
    # Given a detection result, sort the detections based on the number of detection
    n_detections = len(result.boxes)
    detections = torch.zeros((n_target, 4), device=device)
    detect_mask = torch.zeros((n_target), dtype=torch.bool, device=device)
    # print('n_detections', n_detections)

    if n_detections == 0:
        #print('0 detections found')
        return detections, detect_mask
    
    cls_index = result.boxes.cls.int()  # Get the class indices of the detections
    if n_detections <= n_target:
        # Fill in the detecting tensor with the xywhn values
        detections[cls_index] = result.boxes.xywhn  # Fill the detections tensor with the xywhn values
        detect_mask[cls_index] = 1  # Mark the detected targets

    else: # Redundant detections exist
        # The output boxes are automatically sorted by confidence score in ultralytics YOLO
        # Fill the detections tensor with the sorted boxes
        #print(n_detections, 'detections found, but only', n_target, 'targets are needed')
        for cls_id in range(n_target):
            # Get indices where cls_id is found in sorted_cls
            indices = torch.nonzero(cls_index == cls_id, as_tuple=False)

            # Get the first index if it exists, and move it to CPU
            first_index = indices[0].item() if indices.numel() > 0 else -1
            #print('detections for cls_id', cls_id, 'first_index', first_index)

            # Fill the detections tensor with the first detection of the cls_id
            if first_index > -1:
                detections[cls_id] = result.boxes.xywhn[first_index]
                detect_mask[cls_id] = 1  # Mark the detected targets
    
    # print(detections)

    return detections, detect_mask


def load_yolo_model(yolo_model_name, device):
    # Load the YOLO model
    # Set the shared instance of YOLO model in each client model and global
    yolo_model = YOLO(yolo_model_name)
    yolo_model.eval()
    yolo_model = yolo_model.to(device)
    # Freeze YOLO model
    for param in yolo_model.parameters():
        param.requires_grad = False

    return yolo_model


def main():
    rospy.init_node("data_logger", anonymous=True)

    # Load yolo model
    global yolo_model
    global yolo_model_name
    yolo_model = load_yolo_model(yolo_model_name, device)

    # Initialize CSV logs
    init_csv_odom_file(log_file_odom)
    init_csv_odom_file(log_file_ground_truth)
    init_csv_odom_file(log_file_target_1)
    init_csv_odom_file(log_file_target_2)
    init_csv_odom_file(log_file_target_3)

    # Use message_filters to subscribe to multiple topics
    odom_sub = message_filters.Subscriber("/drone1/odometry_sensor1/odometry", Odometry)
    ground_truth_sub = message_filters.Subscriber("/drone1/ground_truth/odometry", Odometry)
    car_sub1 = message_filters.Subscriber("/car_1/odometry", Odometry)
    car_sub2 = message_filters.Subscriber("/car_2/odometry", Odometry)
    car_sub3 = message_filters.Subscriber("/car_3/odometry", Odometry)
    image_sub = message_filters.Subscriber("/drone1/rgb_camera/rgb_camera/image_raw", Image)

    # Synchronize messages
    sync = message_filters.ApproximateTimeSynchronizer([odom_sub, ground_truth_sub, 
                                                        car_sub1, car_sub2, car_sub3, image_sub], 
                                                        queue_size=50,
                                                        slop=0.05)  # Adjust the slop as needed
    sync.registerCallback(synchronized_callback)

    # Use threading to prevent blocking
    spin_thread = threading.Thread(target=rospy.spin)
    spin_thread.start()

    #rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
