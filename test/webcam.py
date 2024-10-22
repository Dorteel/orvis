#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class WebcamPublisher:
    def __init__(self):
        self.bridge = CvBridge()
        self.pub = rospy.Publisher("/webcam/image_raw", Image, queue_size=10)
        self.cap = cv2.VideoCapture(0)  # Open the default webcam

    def start_publishing(self):
        rospy.loginfo("Starting webcam publisher...")
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                rospy.logwarn("Failed to capture frame from webcam.")
                continue

            # Convert the frame to a ROS Image message
            ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.pub.publish(ros_image)
            rate.sleep()

if __name__ == "__main__":
    rospy.init_node('webcam_publisher')
    publisher = WebcamPublisher()
    publisher.start_publishing()
