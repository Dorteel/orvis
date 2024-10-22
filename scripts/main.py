#!/home/user/pel_ws/pel_venv/bin/python

import rospy
import yaml
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import rospkg
from orvis.srv import ImageDetection, ImageDetectionRequest, ImageDetectionResponse  # Updated service and message types
from orvis.msg import BoundingBoxes, BoundingBox  # Import the message types

class MainAnnotatorClient:
    def __init__(self):
        # Initialize the node
        rospy.init_node('main_annotator_client')

        # Define the rate (X seconds)
        self.request_interval = rospy.Duration(2.0)  # Adjust X seconds here (e.g., 2.0 for every 2 seconds)
        self.last_request_time = rospy.Time.now()  # To keep track of the last time the request was sent


        # Define the service we want to test
        service_to_test = '/detr_resnet_50/detect'
        service_type = ImageDetection

        # Initialize the CvBridge
        self.bridge = CvBridge()

        # Use rospkg to get the path of the "orvis" package
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('orvis')

        # Load the main config file
        main_config_path = f"{package_path}/config/main_config.yaml"
        with open(main_config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # Determine logging level
        self.logging_level = self.config['system']['logging_level']

        # Create a subscriber for the camera topic
        self.image_sub = rospy.Subscriber('/webcam/image_raw', Image, self.image_callback)

        # Set up the annotator service client
        rospy.wait_for_service(service_to_test)  # Change to your actual service name
        self.annotator_service = rospy.ServiceProxy(service_to_test, service_type)


        rospy.loginfo("MainAnnotatorClient initialized.")

    def image_callback(self, img_msg):
        """Callback function for the image topic."""

        # Check if enough time has passed since the last request
        current_time = rospy.Time.now()
        if (current_time - self.last_request_time) < self.request_interval:
            return  # Skip this image if X seconds have not passed yet

        # Update the last request time
        self.last_request_time = current_time

        try:
            # Convert the ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")

            # Send the image to the annotator service
            request = ImageDetectionRequest()
            request.image = img_msg  # Assuming the service accepts an image in the request
            response = self.annotator_service(request)

            # If logging level is DEBUG, display the bounding boxes
            if self.logging_level == 'DEBUG':
                self.display_bounding_boxes(cv_image, response)
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def display_bounding_boxes(self, image, response):
        """Display the bounding boxes from the annotator service response."""
        for bbox in response.objects.bounding_boxes:
            # bbox contains xmin, ymin, xmax, ymax, Class, probability
            x_min = int(bbox.xmin)
            y_min = int(bbox.ymin)
            x_max = int(bbox.xmax)
            y_max = int(bbox.ymax)

            # Draw the bounding box on the image
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, f"{bbox.Class} ({round(bbox.probability, 2)})", 
                        (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the image with bounding boxes
        cv2.imshow('Annotated Image', image)
        cv2.waitKey(1)

if __name__ == "__main__":
    try:
        annotator_client = MainAnnotatorClient()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
