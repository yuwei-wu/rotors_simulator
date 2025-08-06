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
import torch
import threading
from torchvision import transforms
from PIL import Image as PILImage
from std_msgs.msg import Float32MultiArray

from model_utils import load_yolo_model, yolo_detect
from model_utils import load_traj_model, traj_pred

# File for logging
log_file_odom = "./drone_log.csv"
log_file_ground_truth = "./drone_ground_truth.csv"
log_file_target_1 = "./target_1_log.csv"
log_file_target_2 = "./target_2_log.csv"
log_file_target_3 = "./target_3_log.csv"
log_file_bbox = "./bbox_log.csv"
log_image_dir = "./drone_images"
last_time = 0

yolo_model_name = "/home/maokat/Documents/fed_learn_quad/catkin_ws/src/rotors_simulator/rotors_gazebo/scripts/best_yolo.pt"
traj_model_name = "/home/maokat/Documents/fed_learn_quad/catkin_ws/src/rotors_simulator/rotors_gazebo/scripts/best_model.pth"

transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize to model's input
    transforms.ToTensor(),  # Convert to tensor (already done in dataset)
])

class DroneProcessor:
    def __init__(self):
        try:
        
            #setup threading lock
            self.lock = threading.Lock()

            #get parameters
            self.drone_id = rospy.get_param('~drone_id', 1)
            self.target_num = rospy.get_param('~target_num', 3)
            self.win_size = rospy.get_param('~win_size', 40)
            self.pred_win_size = rospy.get_param('~pred_win_size', 20)

            #initialize models
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.yolo_model = load_yolo_model(yolo_model_name, self.device)
            self.traj_model = load_traj_model(self.win_size, self.pred_win_size, 
                                            self.target_num, self.device, traj_model_name)
            
            #initialize odom/bbox data arrays
            self.odom_data = torch.zeros((1, self.win_size * 12), dtype=torch.float32)
            self.bbox_data = torch.zeros((1, self.target_num, self.win_size * 4), dtype=torch.float32)

            # Setup publishers
            self.bbox_pub = rospy.Publisher(f"drone{self.drone_id}/target_bbox", Float32MultiArray, queue_size=10)

            # Setup subscribers
            self.setup_subscribers()

            # Setup Logging
            self.log_dir = f"./drone_{self.drone_id}_logs"
            # self.setup_logging()

            rospy.loginfo(f"Drone {self.drone_id} processor initialized")
            
        except Exception as e:
            rospy.logerr(f"Initialization failed: {str(e)}")
            import traceback
            rospy.logerr(traceback.format_exc())
            raise  # Re-raise to see error in terminal

    def setup_subscribers(self):
        # Drone-specific topics
        odom_topic = f"/drone{self.drone_id}/odometry_sensor{self.drone_id}/odometry"
        ground_truth_topic = f"/drone{self.drone_id}/ground_truth/odometry"
        image_topic = f"/drone{self.drone_id}/rgb_camera/rgb_camera/image_raw"
        
        # Shared car topics
        car_topics = [f"/car_{i}/odometry" for i in range(1, self.target_num+1)]
        
        # Create subscribers
        odom_sub = message_filters.Subscriber(odom_topic, Odometry)
        ground_truth_sub = message_filters.Subscriber(ground_truth_topic, Odometry)
        image_sub = message_filters.Subscriber(image_topic, Image)
        car_subs = [message_filters.Subscriber(topic, Odometry) for topic in car_topics]
        
        # Synchronize
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [odom_sub, ground_truth_sub, *car_subs, image_sub],
            queue_size=50,
            slop=0.1
        )
        self.ts.registerCallback(self.synchronized_callback)

    def setup_logging(self):
        # Create log directory
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        # Initialize CSV files
        self.init_csv_file(f"{self.log_dir}/odom.csv", "odom")
        self.init_csv_file(f"{self.log_dir}/ground_truth.csv", "odom")
        for i in range(1, 4):
            self.init_csv_file(f"{self.log_dir}/target_{i}.csv", "odom")
        self.init_csv_file(f"{self.log_dir}/bbox.csv", "bbox", self.target_num)
        
        # Image directory
        self.image_dir = f"{self.log_dir}/images"
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

    def init_csv_file(self, filename, type_, target_num=None):
        with open(filename, "w") as file:
            writer = csv.writer(file)
            if type_ == "odom":
                writer.writerow(["Timestamp", "X", "Y", "Z", "Roll", "Pitch", "Yaw",
                               "Linear_Vel_X", "Linear_Vel_Y", "Linear_Vel_Z",
                               "Angular_Vel_X", "Angular_Vel_Y", "Angular_Vel_Z"])
            elif type_ == "bbox":
                header = ['timestamp']
                for i in range(target_num):
                    header.extend([f'target_{i}_x', f'target_{i}_y', f'target_{i}_w', f'target_{i}_h'])
                writer.writerow(header)

    def process_odom(self, timestamp, msg, log_filename):
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

        # # Log data
        data = [timestamp, x, y, z, roll, pitch, yaw,
                linear_velocity.x, linear_velocity.y, linear_velocity.z,
                angular_velocity.x, angular_velocity.y, angular_velocity.z]
        # with open(log_filename, "a") as file:
        #     writer = csv.writer(file)
        #     writer.writerow(data)

        # rospy.loginfo(f"Drone {self.drone_id} processed odometry at {timestamp}: {data}")

        return data[1:] # Return position and orientation for further processing if needed

    def process_image(self, image_msg, timestamp):
        #TODO - this seems to have issues when no targets are within field? but only for drone 1? little confusing
        bridge = CvBridge()
        try:
            # Convert the ROS image message to an OpenCV image
            cv_image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
            image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(image_rgb)

            image_tensor = transform(pil_image)

            # Add batch dimension and move to device
            image_tensor = image_tensor.unsqueeze(0).to(self.device)  # Shape: (1, 3, 640, 640)
            # Perform YOLO detection
            img_xywhn, detect_masks = yolo_detect(self.yolo_model, image_tensor, self.target_num, self.device)
            self.bbox_data = torch.cat((self.bbox_data[:, :, :(self.win_size-1)*4], img_xywhn.cpu()), dim=2)

            # Perform trajectory prediction
            pred_traj = traj_pred(self.traj_model, self.odom_data, self.bbox_data, 
                                  self.target_num, self.win_size, self.pred_win_size, self.device)

            # Save the image
            # Get the current timestamp for unique filenames
            # img_filename = os.path.join(log_image_dir, "{:.6f}.png".format(timestamp))
            # cv2.imwrite(img_filename, cv_image)

            # rospy.loginfo(f"Drone {self.drone_id} processed image at {timestamp}, saved to {img_filename}")

        except Exception as e:
            rospy.logerr(f"Drone {self.drone_id}: Failed to convert image: {e}")
            return None, None, None

        return img_xywhn, pred_traj
    
    def process_bbox(self, timestamp, bbox, pred_traj, log_filename):
        # Log bounding box data
        with open(log_filename, "a") as file:
            writer = csv.writer(file)
            row = [timestamp] + bbox.tolist()
            writer.writerow(row)
        bbox = bbox.tolist()

        data = pred_traj.flatten().tolist()
        #packet = [predicted window size, target num, 1 if target is real, else 0, predicted trajectory points...]
        packet = [self.pred_win_size, self.target_num]
        traj_packet = []
        for i in range(self.target_num): #check if target is real
            if all(b == 0 for b in bbox[i*4:i*4+4]): #if all bbox values are zero, target not real
                packet.append(0)
            else: #target is real, sending pred traj info
                packet.append(1)
                traj_packet.extend(data[i*self.pred_win_size*2:(i+1)*self.pred_win_size*2]) #flattened pred traj for target i
        packet.extend(traj_packet) #add flattened pred traj to packet

        # print('packet', packet)

        pred_traj_msg = Float32MultiArray(data=packet) #TODO - check this
        self.bbox_pub.publish(pred_traj_msg)
        # rospy.loginfo(f"Drone {self.drone_id} published predicted trajectory at {timestamp}")
    
    def synchronized_callback(self, odom_msg, ground_truth_msg, *car_msgs):
        image_msg = car_msgs[-1]  # The last message is the image
        car_msgs = car_msgs[:-1]  # All but the last are car odometry

        rospy.loginfo(f"Drone {self.drone_id} received synchronized messages at {rospy.get_time()}")
    
        with self.lock: #ensure thread saftey
            timestamp = rospy.get_time()
            # try:
            #process odom
            odom = self.process_odom(timestamp, odom_msg, f"{self.log_dir}/odom.csv")
            odom = torch.tensor(np.array(odom).reshape(1, -1), dtype=torch.float32)   
            self.odom_data = torch.cat((self.odom_data[:, :(self.win_size-1)*12], odom), dim=1)

            #process ground truth
            self.process_odom(timestamp, ground_truth_msg, f"{self.log_dir}/ground_truth.csv")

            #process car odom   
            for i, car_msg in enumerate(car_msgs, 1):
                self.process_odom(timestamp, car_msg, f"{self.log_dir}/target_{i}.csv")

            #process image
            img_xywhn, pred_traj = self.process_image(image_msg, timestamp)

            # # Log bounding boxes
            # bbox = img_xywhn.cpu().numpy().reshape(-1)
            # self.process_bbox(timestamp, bbox, pred_traj, log_file_bbox) #all three targets, zeros out if less than target num

            # except Exception as e:
            #     rospy.logerr(f"Processing error in drone {self.drone_id}: {str(e)}")
            #     return

 

if __name__ == "__main__":
    rospy.init_node("drone_processor")
    try:
        processor = DroneProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
