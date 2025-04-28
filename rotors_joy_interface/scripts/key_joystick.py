#!/usr/bin/env python3
"""
Virtual Joystick from Keyboard → ROS sensor_msgs/Joy publisher
Adapted from Bharat Tak's original uinput-based key_joystick.py
"""

import os
import pygame
import rospy
from pygame.locals import *
from sensor_msgs.msg import Joy

# Initialize pygame and window
pygame.init()
WHITE = (250, 250, 250)
WIDTH, HEIGHT = 485, 530
windowSurface = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
pygame.display.set_caption('Position Controller Joystick')

dir_path = os.path.dirname(__file__)
# Pre-render background and static overlay
background = pygame.Surface((WIDTH, HEIGHT)).convert()
background.fill(WHITE)
sticks_img = pygame.image.load(os.path.join(dir_path, '../media/sticks.png')).convert_alpha()

class stick_state(object):
    def __init__(self, name, key_up, key_down, display_h, display_w, horizontal=True, spring_back=True, incr_val=0.1):
        self.name = name
        self.key_up = key_up
        self.key_down = key_down
        self.spring_back = spring_back
        self.incr_val = incr_val
        self.min_val = 0.0
        self.max_val = 255.0
        self.zero = 127.0 if spring_back else 0.0
        self.val = self.zero
        self.emit_val = int(self.zero)
        # display params
        self.display_h = display_h
        self.display_w = display_w
        self.display_hor = horizontal
        # load bar images
        if horizontal:
            self.bar_g = pygame.image.load(os.path.join(dir_path, '../media/hg.png')).convert_alpha()
            self.bar_b = pygame.image.load(os.path.join(dir_path, '../media/hb.png')).convert_alpha()
        else:
            self.bar_g = pygame.image.load(os.path.join(dir_path, '../media/vg.png')).convert_alpha()
            self.bar_b = pygame.image.load(os.path.join(dir_path, '../media/vb.png')).convert_alpha()
        self.active_up = False
        self.active_down = False

    def update_event(self, event):
        if event.type == KEYDOWN:
            if event.key == self.key_up:
                self.active_up = True
            elif event.key == self.key_down:
                self.active_down = True
        elif event.type == KEYUP:
            if event.key == self.key_up:
                self.active_up = False
            elif event.key == self.key_down:
                self.active_down = False

    def update_value(self):
        if self.active_up:
            self.val = min(self.val + self.incr_val, self.max_val)
        elif self.active_down:
            self.val = max(self.val - self.incr_val, self.min_val)
        else:
            if self.spring_back:
                if self.val > self.zero:
                    self.val -= self.incr_val * 0.2
                elif self.val < self.zero:
                    self.val += self.incr_val * 0.2
                else:
                    self.val = self.zero
        self.emit_val = int(round(self.val))

    def draw(self, surface):
        # draw bar segments for this stick
        for i in range(256):
            img = self.bar_g if i <= self.emit_val else self.bar_b
            if self.display_hor:
                surface.blit(img, (self.display_w + i, self.display_h))
            else:
                surface.blit(img, (self.display_w, self.display_h - i))

    def get_axis(self):
        # Normalize to [-1.0, 1.0]
        return (self.emit_val - self.zero) / self.zero if self.zero else 0.0


def main():
    rospy.init_node('key_joy_publisher')
    joy_pub = rospy.Publisher('/joy', Joy, queue_size=10)
    rate = rospy.Rate(50)  # update at 50 Hz to reduce CPU load

    # Define sticks with display positions
    sticks = [
        stick_state('X', K_UP, K_DOWN, 320, 90, horizontal=False),
        stick_state('Y', K_LEFT, K_RIGHT, 320, 180, horizontal=False),
        stick_state('Z', K_w, K_s, 320, 270, horizontal=False),
        stick_state('Yaw', K_d, K_a, 320, 360, horizontal=False),
        stick_state('extX', K_u, K_y, 410, 120, horizontal=True),
        stick_state('extY', K_j, K_h, 450, 120, horizontal=True),
        stick_state('extZ', K_m, K_n, 490, 120, horizontal=True),
    ]

    try:
        while not rospy.is_shutdown():
            # poll events
            for event in pygame.event.get():
                if event.type == QUIT:
                    rospy.signal_shutdown('Window closed')
                for s in sticks:
                    s.update_event(event)

            # update stick values
            for s in sticks:
                s.update_value()

            # build and publish Joy msg
            joy = Joy()
            joy.header.stamp = rospy.Time.now()
            joy.axes = [s.get_axis() for s in sticks]
            joy.buttons = []
            joy_pub.publish(joy)

            # redraw entire display once
            windowSurface.blit(background, (0, 0))
            windowSurface.blit(sticks_img, (0, 0))
            for s in sticks:
                s.draw(windowSurface)
            pygame.display.flip()

            rate.sleep()
    except rospy.ROSInterruptException:
        pass
    finally:
        pygame.quit()

if __name__ == '__main__':
    main()
