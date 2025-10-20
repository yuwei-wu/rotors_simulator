#!/usr/bin/env python3
'''
Virtual Joystick from Keyboard - Fixed for Hummingbird Control
Publishes PoseStamped messages to control the drone directly
'''

import os
import time
import pygame
import sys
import rospy
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from std_msgs.msg import Header
import tf.transformations as tf_trans

# Set SDL video driver and display settings for better GUI compatibility
os.environ['SDL_VIDEODRIVER'] = 'x11'
os.environ['DISPLAY'] = ':0'

# Initialize pygame first
pygame.init()

# Import pygame constants after initialization
QUIT = pygame.QUIT
KEYDOWN = pygame.KEYDOWN 
KEYUP = pygame.KEYUP
K_UP = pygame.K_UP
K_DOWN = pygame.K_DOWN
K_LEFT = pygame.K_LEFT
K_RIGHT = pygame.K_RIGHT
K_w = pygame.K_w
K_s = pygame.K_s
K_a = pygame.K_a
K_d = pygame.K_d
K_u = pygame.K_u
K_y = pygame.K_y
K_j = pygame.K_j
K_h = pygame.K_h
K_m = pygame.K_m
K_n = pygame.K_n
K_ESCAPE = pygame.K_ESCAPE

# 初始化 pygame 窗口
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WIDTH = 600
HEIGHT = 400

try:
    windowSurface = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
    windowSurface.fill(WHITE)
    pygame.display.set_caption('Hummingbird Position Controller')
    print("✅ Pygame window created successfully")
except Exception as e:
    print(f"❌ Error creating pygame window: {e}")
    sys.exit(1)

class DroneController(object):
    def __init__(self, name, key_up, key_down, increment=0.1):
        self.name = name
        self.key_up = key_up
        self.key_down = key_down
        self.increment = increment
        self.value = 0.0
        self.active_up = False
        self.active_down = False
        
        # Set limits based on control type
        if name in ['X', 'Y']:
            self.min_val = -500.0  # X and Y position limits in meters
            self.max_val = 500.0
        elif name == 'Z':
            self.min_val = 0.0   # Z position limit in meters
            self.max_val = 50.0
        else:  # Yaw
            self.min_val = float('-inf')  # Unlimited yaw rotation
            self.max_val = float('inf')
            
    def keypress_up(self):
        self.active_up = True
        new_val = self.value + self.increment
        self.value = min(new_val, self.max_val)
        
    def keypress_down(self):
        self.active_down = True
        new_val = self.value - self.increment
        self.value = max(new_val, self.min_val)
        
    def update_event(self, event):
        if event.type == KEYDOWN:
            if event.key == self.key_up:
                self.keypress_up()
            elif event.key == self.key_down:
                self.keypress_down()
        elif event.type == KEYUP:
            if event.key == self.key_up:
                self.active_up = False
            elif event.key == self.key_down:
                self.active_down = False
                
    def update(self):
        if self.active_up:
            self.keypress_up()
        elif self.active_down:
            self.keypress_down()
        return self.value

def main():
    # 初始化 ROS 节点和发布者
    rospy.init_node('virtual_joystick', anonymous=True)
    # 发布到 /hummingbird/command/pose 话题，消息类型 geometry_msgs/PoseStamped
    pub = rospy.Publisher('/hummingbird/command/pose', PoseStamped, queue_size=10)

    # 创建各个控制轴，使用drone-relative控制：
    # forward/back: K_UP / K_DOWN (relative to drone's front)
    forward_controller = DroneController('Forward', K_UP, K_DOWN, increment=0.1)
    # left/right: K_LEFT / K_RIGHT (relative to drone's sides)  
    right_controller = DroneController('Right', K_RIGHT, K_LEFT, increment=0.1)
    # z: K_w / K_s (altitude)
    z_controller = DroneController('Z', K_w, K_s, increment=0.1)
    # yaw: K_d / K_a (slower rotation)
    yaw_controller = DroneController('Yaw', K_d, K_a, increment=0.03)

    controllers = [forward_controller, right_controller, z_controller, yaw_controller]

    rate = rospy.Rate(50)  # 50 Hz 发布频率
    
    # Initialize font
    pygame.font.init()
    font = pygame.font.Font(None, 24)
    
    print("🎮 Virtual Joystick Started!")
    print("📡 Publishing to /hummingbird/command/pose")
    print("🎯 Use arrow keys, WASD, and AD to control the drone")

    while not rospy.is_shutdown():
        # 处理 pygame 事件
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                print("👋 Exiting...")
                pygame.quit()
                sys.exit()
            for controller in controllers:
                controller.update_event(event)

        # 更新所有控制的当前数值
        for controller in controllers:
            controller.update()

        # 可选：在界面上刷新显示当前原始值
        windowSurface.fill(WHITE)
        
        # Display current values
        for idx, controller in enumerate(controllers):
            text_str = f"{controller.name}: {controller.value:.2f}"
            color = GREEN if abs(controller.value) > 0.01 else BLACK
            text = font.render(text_str, True, color)
            windowSurface.blit(text, (10, 10 + idx*30))
            
        # Add instructions
        instructions = [
            "🎮 Hummingbird Drone Control",
            "",
            "📍 Drone-Relative Control:",
            "  Forward: ↑/↓ arrows (drone front/back)",
            "  Right: ←/→ arrows (drone left/right)", 
            "  Z: W/S keys (up/down)",
            "",
            "🔄 Orientation:", 
            "  Yaw: A/D keys (rotate unlimited)",
            "",
            "🔧 Controls:",
            "  ESC: Exit",
            "",
            "📡 Status: Publishing to ROS ✅" if not rospy.is_shutdown() else "📡 Status: ROS Disconnected ❌"
        ]
        
        for idx, instruction in enumerate(instructions):
            color = BLUE if instruction.startswith("🎮") else BLACK
            if instruction.startswith("📡 Status") and "✅" in instruction:
                color = GREEN
            elif instruction.startswith("📡 Status") and "❌" in instruction:
                color = RED
                
            text = font.render(instruction, True, color)
            windowSurface.blit(text, (250, 10 + idx*20))
            
        pygame.display.flip()

        # 构造 geometry_msgs/PoseStamped 消息并发布
        pose_msg = PoseStamped()
        pose_msg.header = Header()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "world"
        
        # Convert drone-relative movement to world coordinates
        import math
        current_yaw = yaw_controller.value
        forward_distance = forward_controller.value
        right_distance = right_controller.value
        
        # Transform to world coordinates based on current yaw
        world_x = forward_distance * math.cos(current_yaw) - right_distance * math.sin(current_yaw)
        world_y = forward_distance * math.sin(current_yaw) + right_distance * math.cos(current_yaw)
        
        # 设置位置 (X, Y, Z)
        pose_msg.pose.position = Point()
        pose_msg.pose.position.x = world_x
        pose_msg.pose.position.y = world_y  
        pose_msg.pose.position.z = z_controller.value
        
        # 设置姿态 (Yaw)
        quaternion = tf_trans.quaternion_from_euler(0, 0, yaw_controller.value)
        pose_msg.pose.orientation = Quaternion()
        pose_msg.pose.orientation.x = quaternion[0]
        pose_msg.pose.orientation.y = quaternion[1]
        pose_msg.pose.orientation.z = quaternion[2]
        pose_msg.pose.orientation.w = quaternion[3]
        
        pub.publish(pose_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        print("🔴 ROS Interrupted")
        pass
    except KeyboardInterrupt:
        print("🔴 Keyboard Interrupt")
        pass
    finally:
        pygame.quit()
