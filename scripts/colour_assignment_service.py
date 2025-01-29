import rospy
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from orvis.srv import AssignColour, AssignColourResponse
from cv_bridge import CvBridge
from sklearn.cluster import KMeans

class ColorAssignerService:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('assign_color_service')

        # ROS Parameters
        self.method = rospy.get_param('~method', 'most_common')  # 'most_common' or 'average'
        self.window_size = rospy.get_param('~window_size', 9)  # Size of the bounding box window
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

            # Extract pixels based on the input type
            if req.input_type == 'mask':
                mask = self.bridge.imgmsg_to_cv2(req.mask, desired_encoding='mono8')
                pixels = cv_image[mask > 0]
            elif req.input_type == 'bounding_box':
                x_center, y_center = req.bounding_box.center.x, req.bounding_box.center.y
                half_window = self.window_size // 2

                # Crop the bounding box window
                x_min = max(0, x_center - half_window)
                y_min = max(0, y_center - half_window)
                x_max = min(cv_image.shape[1], x_center + half_window)
                y_max = min(cv_image.shape[0], y_center + half_window)

                pixels = cv_image[y_min:y_max, x_min:x_max].reshape(-1, 3)
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
        return "#{:02x}{:02x}{:02x}".format(color[2], color[1], color[0])  # BGR to RGB

if __name__ == '__main__':
    try:
        ColorAssignerService()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.logerr("Color Assigner Service terminated.")
