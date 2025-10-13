#!/usr/bin/env python3
'''
Virtual Joystick from Keyboard without uinput
重构后的虚拟摇杆：剥离 uinput，直接发布 sensor_msgs/Joy 消息
'''

import os
import time
import pygame, sys
import rospy
from pygame.locals import *
from sensor_msgs.msg import Joy

# 初始化 pygame 窗口
pygame.init()
WHITE = (255, 255, 255)
WIDTH = 485
HEIGHT = 530
windowSurface = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
windowSurface.fill(WHITE)
pygame.display.set_caption('Position Controller Joystick')

# 加载背景图片（如果需要）
dir_path = os.path.dirname(__file__)
bg_img_path = os.path.join(dir_path, '../media/sticks.png')
if os.path.exists(bg_img_path):
    img = pygame.image.load(bg_img_path)
    windowSurface.blit(img, (0, 0))
pygame.display.flip()

class StickState(object):
    def __init__(self, name, key_up, key_down, spring_back=True, incr_val=1.0):
        self.name = name                # 控制名称
        self.key_up = key_up            # 增加对应键
        self.key_down = key_down        # 减少对应键
        self.spring_back = spring_back  # 松开后是否回弹到中心
        self.incr_val = incr_val        # 每次按键的增量
        self.min_val = 0.0              # 最小值
        self.max_val = 255.0            # 最大值
        # 对于有回弹的摇杆，中心值设为127；否则初始为0
        self.zero = 127.0 if spring_back else 0.0
        self.val = self.zero
        self.active_up = False
        self.active_down = False

    def keypress_up(self):
        self.active_up = True
        if self.val + self.incr_val <= self.max_val:
            self.val += self.incr_val
        else:
            self.val = self.max_val

    def keypress_down(self):
        self.active_down = True
        if self.val - self.incr_val >= self.min_val:
            self.val -= self.incr_val
        else:
            self.val = self.min_val

    def release_stick(self):
        if not self.spring_back:
            return
        # 回弹到中心（逐渐恢复）
        if self.val > self.zero:
            self.val -= self.incr_val * 0.2
            if self.val < self.zero:
                self.val = self.zero
        elif self.val < self.zero:
            self.val += self.incr_val * 0.2
            if self.val > self.zero:
                self.val = self.zero

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
        else:
            self.release_stick()
        # 限制数值范围
        if self.val < self.min_val:
            self.val = self.min_val
        if self.val > self.max_val:
            self.val = self.max_val
        return self.val

    def get_normalized(self):
        """
        对于有回弹的轴，将 [0,255] 转换为 [-1,1]（中心 127 对应 0）；
        对于非回弹轴，归一化到 [0,1]
        """
        if self.spring_back:
            return (self.val - self.zero) / self.zero
        else:
            return self.val / self.max_val

def main():
    # 初始化 ROS 节点和发布者
    rospy.init_node('virtual_joystick', anonymous=True)
    # 发布到 /hummingbird/joy 话题，消息类型 sensor_msgs/Joy
    pub = rospy.Publisher('/hummingbird/joy', Joy, queue_size=10)

    # 创建各个控制轴，按键映射与原代码一致：
    # x: K_UP / K_DOWN
    x_stick = StickState('X', K_UP, K_DOWN)
    # y: K_LEFT / K_RIGHT
    y_stick = StickState('Y', K_LEFT, K_RIGHT)
    # z: K_w / K_s
    z_stick = StickState('Z', K_w, K_s)
    # yaw: K_d / K_a
    yaw_stick = StickState('Yaw', K_d, K_a)
    # 外部扰动力（若需要）：extforceX: K_u / K_y
    extforce_x = StickState('extforceX', K_u, K_y, spring_back=False, incr_val=5.0)
    # extforceY: K_j / K_h
    extforce_y = StickState('extforceY', K_j, K_h, spring_back=False, incr_val=5.0)
    # extforceZ: K_m / K_n
    extforce_z = StickState('extforceZ', K_m, K_n, spring_back=False, incr_val=5.0)

    sticks = [x_stick, y_stick, z_stick, yaw_stick, extforce_x, extforce_y, extforce_z]

    rate = rospy.Rate(50)  # 50 Hz 发布频率

    while not rospy.is_shutdown():
        # 处理 pygame 事件
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            for stick in sticks:
                stick.update_event(event)

        # 更新所有控制的当前数值
        raw_values = [stick.update() for stick in sticks]
        norm_values = [stick.get_normalized() for stick in sticks]

        # 可选：在界面上刷新显示当前原始值
        windowSurface.fill(WHITE)
        font = pygame.font.SysFont("Arial", 16)
        for idx, stick in enumerate(sticks):
            text = font.render(f"{stick.name}: {int(stick.val)}", True, (0,0,0))
            windowSurface.blit(text, (10, 10 + idx*20))
        pygame.display.flip()

        # 构造 sensor_msgs/Joy 消息并发布
        joy_msg = Joy()
        joy_msg.header.stamp = rospy.Time.now()
        # 轴顺序：X, Y, Z, Yaw, extforceX, extforceY, extforceZ
        joy_msg.axes = norm_values
        # 如果没有按钮状态，可以为空
        joy_msg.buttons = []
        pub.publish(joy_msg)

        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
