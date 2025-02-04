#!/home/user/pel_ws/pel_venv/bin/python

import rospy
import numpy as np
from sensor_msgs.msg import Image
from orvis.srv import AssignColour, AssignColourResponse
from cv_bridge import CvBridge
from sklearn.cluster import KMeans
import cv2

class ColorAssignerService:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('assign_color_service')

        # ROS Parameters
        self.method = rospy.get_param('~method', 'most_common')  # 'most_common' or 'average'
        self.num_clusters = rospy.get_param('~num_clusters', 5)  # Number of clusters for k-means

        # Service setup
        self.service = rospy.Service('/assign_color', AssignColour, self.handle_assign_color)

        # CV Bridge for image conversion
        self.bridge = CvBridge()

        rospy.loginfo("Color Assigner Service is ready.")

    def handle_assign_color(self, req):
        try:
            # Convert image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(req.image, desired_encoding='bgr8')
            
            # Extract pixels based on input type
            if req.input_type == 'mask':
                mask_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                _, mask_binary = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)
                pixels = cv_image[mask_binary > 0]
            elif req.input_type == 'bounding_box':
                pixels = cv_image.reshape(-1, 3)  # Use the whole image
            else:
                return AssignColourResponse(success=False, hex_color="", message="Invalid input type")

            # Check if pixels are valid
            if len(pixels) == 0:
                return AssignColourResponse(success=False, hex_color="", message="No valid pixels found")

            # Determine the color based on the method
            if self.method == 'most_common':
                hex_color = self.get_most_common_color(pixels)
            elif self.method == 'average':
                hex_color = self.get_average_color(pixels)
            else:
                return AssignColourResponse(success=False, hex_color="", message="Invalid method")

            return AssignColourResponse(success=True, hex_color=hex_color, message="Color assigned successfully")
        except Exception as e:
            rospy.logerr(f"Error in handle_assign_color: {e}")
            return AssignColourResponse(success=False, hex_color="", message=str(e))

    def get_most_common_color(self, pixels):
        # Use k-means clustering to find the most common color
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42).fit(pixels)
        most_common_cluster = np.argmax(np.bincount(kmeans.labels_))
        most_common_color = kmeans.cluster_centers_[most_common_cluster].astype(int)
        return self.rgb_to_hex(most_common_color)

    def get_average_color(self, pixels):
        # Calculate the average color
        average_color = np.mean(pixels, axis=0).astype(int)
        return self.rgb_to_hex(average_color)

    @staticmethod
    def rgb_to_hex(color):
        # Convert RGB to hex
        return "{:02x}{:02x}{:02x}".format(color[2], color[1], color[0])  # BGR to RGB

if __name__ == '__main__':
    try:
        ColorAssignerService()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.logerr("Color Assigner Service terminated.")
