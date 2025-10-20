#!/usr/bin/env python3
"""
Random Actor Controller
Publishes random velocity commands to make an actor walk around randomly
"""

import rospy
import random
import math
from geometry_msgs.msg import Twist

class RandomActorController:
    def __init__(self):
        rospy.init_node('random_actor_controller', anonymous=True)
        
        # Publisher for actor velocity commands
        self.vel_pub = rospy.Publisher('/random_walker/cmd_vel', Twist, queue_size=10)
        
        # Control parameters
        self.max_linear_vel = 1.5   # Max forward/backward speed
        self.max_angular_vel = 1.0  # Max turning speed
        self.direction_change_interval = 3.0  # Change direction every 3 seconds
        
        # Current state
        self.current_linear = 0.0
        self.current_angular = 0.0
        self.last_change_time = rospy.Time.now()
        
        # Timer for control loop
        self.timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)
        
        rospy.loginfo("🚶 Random Actor Controller Started!")
        rospy.loginfo("📡 Publishing to /random_walker/cmd_vel")

    def control_loop(self, event):
        """Main control loop - generates random walking patterns"""
        current_time = rospy.Time.now()
        
        # Change direction periodically or randomly
        if (current_time - self.last_change_time).to_sec() > self.direction_change_interval:
            self.generate_new_direction()
            self.last_change_time = current_time
        
        # Create and publish velocity command
        twist = Twist()
        twist.linear.x = self.current_linear
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = self.current_angular
        
        self.vel_pub.publish(twist)

    def generate_new_direction(self):
        """Generate a new random walking direction"""
        # Random walking patterns
        patterns = [
            'forward',      # Walk straight forward
            'backward',     # Walk backward
            'turn_left',    # Turn left while walking
            'turn_right',   # Turn right while walking
            'circle_left',  # Walk in a circle (left)
            'circle_right', # Walk in a circle (right)
            'stop',         # Stop for a moment
            'random_walk'   # Random combination
        ]
        
        pattern = random.choice(patterns)
        
        if pattern == 'forward':
            self.current_linear = random.uniform(0.5, self.max_linear_vel)
            self.current_angular = 0.0
            
        elif pattern == 'backward':
            self.current_linear = random.uniform(-0.5, -0.2)
            self.current_angular = 0.0
            
        elif pattern == 'turn_left':
            self.current_linear = random.uniform(0.3, 0.8)
            self.current_angular = random.uniform(0.3, self.max_angular_vel)
            
        elif pattern == 'turn_right':
            self.current_linear = random.uniform(0.3, 0.8)
            self.current_angular = random.uniform(-self.max_angular_vel, -0.3)
            
        elif pattern == 'circle_left':
            self.current_linear = 0.8
            self.current_angular = 0.6
            
        elif pattern == 'circle_right':
            self.current_linear = 0.8
            self.current_angular = -0.6
            
        elif pattern == 'stop':
            self.current_linear = 0.0
            self.current_angular = 0.0
            # Shorter stop duration
            self.direction_change_interval = random.uniform(1.0, 2.0)
            
        else:  # random_walk
            self.current_linear = random.uniform(-0.5, self.max_linear_vel)
            self.current_angular = random.uniform(-self.max_angular_vel, self.max_angular_vel)
        
        # Vary the duration for each direction
        if pattern != 'stop':
            self.direction_change_interval = random.uniform(2.0, 5.0)
        
        rospy.loginfo(f"🎯 New pattern: {pattern} | Linear: {self.current_linear:.2f} | Angular: {self.current_angular:.2f}")

if __name__ == '__main__':
    try:
        controller = RandomActorController()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("🔴 Random Actor Controller Stopped")
