#!/home/user/pel_ws/pel_venv/bin/python

import rospy
import yaml
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import rospkg
from orvis.srv import ImageDetection, ImageDetectionRequest, ImageDetectionResponse  # Detection service
from orvis.srv import ImageMaskDetection, ImageMaskDetectionRequest, ImageMaskDetectionResponse  # Segmentation service
from orvis.msg import ImageMasks, ImageMask  # Import the segmentation message types


class MainAnnotatorClient:
    def __init__(self):
        # Initialize the node
        rospy.init_node('main_annotator_client')

        # Load configuration and determine service type
        self.bridge = CvBridge()
        self.load_config()

        # Define the rate for service requests (e.g., every 2 seconds)
        self.request_interval = rospy.Duration(self.config['system']['request_interval'])
        self.last_request_time = rospy.Time.now()  # Track the last request time

        # Initialize the ROS service
        rospy.wait_for_service(self.service_name)
        self.annotator_service = rospy.ServiceProxy(self.service_name, self.service_type)

        rospy.loginfo(f"Service {self.service_name} connected.")

        # Subscribe to the appropriate image topic
        rospy.Subscriber(self.camera_topic, Image, self.image_callback)
        rospy.loginfo("MainAnnotatorClient initialized.")

    def load_config(self):
        """Load main configuration and determine service parameters."""
        # Use rospkg to get the path of the "orvis" package
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('orvis')
        main_config_path = f"{package_path}/config/main_config.yaml"

        # Load the main config file
        with open(main_config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # Determine service to test (can be adjusted to allow dynamic selection)
        self.service_name = self.config['system']['service_to_test']
        self.task_type = self.config['system']['task_type']
        if self.task_type == 'SegmentationTask':
            self.service_type = ImageMaskDetection 
        elif self.task_type == 'StandardDetectionTask':
            self.service_type = ImageDetection

        # Determine the camera topic
        self.camera_topic = self.config['system']['camera_topic']

        # Determine logging level
        self.logging_level = self.config['system']['logging_level']

    def image_callback(self, img_msg):
        """Callback for the image topic."""
        current_time = rospy.Time.now()
        if (current_time - self.last_request_time) < self.request_interval:
            return  # Skip if the time interval hasn't passed

        # Update the last request time
        self.last_request_time = current_time

        # Send the image to the appropriate service
        if self.service_type == ImageDetection:
            self.process_detection(img_msg)
        elif self.service_type == ImageMaskDetection:
            self.process_segmentation(img_msg)

    def process_detection(self, img_msg):
        """Process image detection service requests."""
        try:
            # Convert the ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            request = ImageDetectionRequest(image=img_msg)
            response = self.annotator_service(request)

            # If logging level is DEBUG, display the bounding boxes
            if self.logging_level == 'DEBUG':
                self.display_bounding_boxes(cv_image, response)
        except Exception as e:
            rospy.logerr(f"Error processing detection image: {e}")

    def process_segmentation(self, img_msg):
        """Process segmentation service requests."""
        try:
            # Convert the ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            request = ImageMaskDetectionRequest(image=img_msg)
            response = self.annotator_service(request)

            # If logging level is DEBUG, display the segmentation masks
            if self.logging_level == 'DEBUG':
                self.display_segmentation_masks(cv_image, response)
        except Exception as e:
            rospy.logerr(f"Error processing segmentation image: {e}")

    def display_bounding_boxes(self, image, response):
        """Display bounding boxes from the detection response."""
        for bbox in response.objects.bounding_boxes:
            x_min, y_min, x_max, y_max = int(bbox.xmin), int(bbox.ymin), int(bbox.xmax), int(bbox.ymax)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, f"{bbox.Class} ({round(bbox.probability, 2)})", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the image
        cv2.imshow('Annotated Image', image)
        cv2.waitKey(1)

    def display_segmentation_masks(self, image, response):
        """Display segmentation masks from the segmentation response."""
        for mask in response.objects.masks:
            # Convert the mask to OpenCV format (mono8)
            mask_image = self.bridge.imgmsg_to_cv2(mask.mask, "mono8")

            # Overlay the mask on the image
            colored_mask = cv2.applyColorMap(mask_image, cv2.COLORMAP_JET)
            overlaid_image = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)

            # Find the top-most pixel in the mask for the label
            mask_indices = cv2.findNonZero(mask_image)
            if mask_indices is not None:
                top_left = tuple(mask_indices.min(axis=0)[0])
                text_position = (top_left[0], max(10, top_left[1] - 10))
                cv2.putText(overlaid_image, f"{mask.Class} ({round(mask.probability, 2)})", text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Display the mask
            cv2.imshow('Segmentation Image', overlaid_image)
            cv2.waitKey(1)


if __name__ == "__main__":
    try:
        annotator_client = MainAnnotatorClient()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
