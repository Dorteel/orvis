#!/home/user/pel_ws/pel_venv/bin/python

import rospy
from sensor_msgs.msg import Image
from orvis.srv import VLM

from cv_bridge import CvBridge

bridge = CvBridge()
IMAGE_TOPIC = "/webcam/image_raw"
SVC_NAME = "/vlm_query_llama-3.2-90b-vision-preview".replace("-", "_").replace(".", "_")

class VLMTester:
    def __init__(self):
        rospy.wait_for_service(SVC_NAME)
        self.query_service = rospy.ServiceProxy(SVC_NAME, VLM)
        rospy.Subscriber(IMAGE_TOPIC, Image, self.image_callback, queue_size=1)
        self.image_sent = False

    def image_callback(self, msg):
        if self.image_sent:
            return
        self.image_sent = True
        try:
            resp = self.query_service(prompt="Describe the image in the for of a scene graph, make it as detailed as possible.", image=msg)
            print("VLM response:", resp.message)
        except rospy.ServiceException as e:
            print("Service call failed:", e)

if __name__ == "__main__":
    rospy.init_node("vlm_test_node")
    tester = VLMTester()
    rospy.spin()
