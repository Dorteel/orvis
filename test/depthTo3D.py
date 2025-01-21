#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

class DepthTo3D:
    def __init__(self):
        rospy.init_node('depth_to_3d', anonymous=True)

        self.bridge = CvBridge()
        self.camera_info = None

        # Subscribers
        rospy.Subscriber('/camera/color/image_raw', Image, self.rgb_callback)
        rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.camera_info_callback)

        self.rgb_image = None
        self.depth_image = None

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def rgb_callback(self, msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def get_3d_coordinates(self, u, v):
        if self.camera_info is None or self.depth_image is None:
            rospy.logwarn("Camera info or depth image not available.")
            return None

        # Intrinsic parameters
        fx = self.camera_info.K[0]
        fy = self.camera_info.K[4]
        cx = self.camera_info.K[2]
        cy = self.camera_info.K[5]

        # Get depth value
        depth = self.depth_image[v, u]

        if depth == 0 or np.isnan(depth):
            rospy.logwarn("Invalid depth value at pixel ({}, {}).".format(u, v))
            return None

        # Reproject to 3D
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        return x, y, z

if __name__ == '__main__':
    depth_to_3d = DepthTo3D()
    rospy.spin()
