#!/home/user/pel_ws/pel_venv/bin/python

import rospy
import yaml
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import rospkg
import numpy as np
from orvis.srv import ObjectDetection, ObjectDetectionRequest, ObjectDetectionResponse  # Detection service
from orvis.srv import ImageSegmentation, ImageSegmentationRequest, ImageSegmentationResponse  # Segmentation service
from orvis.msg import ImageMasks, ImageMask  # Import the segmentation message types
from orvis.srv import PromptedObjectDetection, PromptedObjectDetectionRequest, PromptedObjectDetectionResponse  # Detection service
from orvis.srv import DepthEstimation, DepthEstimationRequest, DepthEstimationResponse  # Import the necessary service types
from orvis.srv import VideoClassification, VideoClassificationRequest, VideoClassificationResponse  # Detection service
from collections import deque
from std_msgs.msg import String

class MainAnnotatorClient:
    def __init__(self):
        # Initialize the node
        rospy.init_node('main_annotator_client')

        # Load configuration and determine service type
        self.bridge = CvBridge()
        self.load_config()
        self.prompt = String()
        self.prompt.data = "pineapple"

        # Define the rate for service requests (e.g., every 2 seconds)
        self.request_interval = rospy.Duration(self.config['system']['request_interval'])
        self.last_request_time = rospy.Time.now()  # Track the last request time

        # Initialize the ROS service
        rospy.wait_for_service(self.service_name)
        self.annotator_service = rospy.ServiceProxy(self.service_name, self.service_type)
        self.video_frames = deque(maxlen=self.num_frames)

        rospy.loginfo(f"Service {self.service_name} connected.")

        # Subscribe to the appropriate image topic
        rospy.Subscriber(self.camera_topic, Image, self.image_callback)
        rospy.loginfo("MainAnnotatorClient initialized.")

    def load_config(self):
        """Load main configuration and determine service parameters."""
        # Use rospkg to get the path of the "orvis" package
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('orvis')
        main_config_path = f"{package_path}/config/orvis_config.yaml"

        # Load the main config file
        with open(main_config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # Determine service to test (can be adjusted to allow dynamic selection)
        self.service_name = self.config['system']['service_to_test']
        self.task_type = self.config['system']['task_type']
        if self.task_type == 'ImageSegmentation':
            self.service_type = ImageSegmentation 
        elif self.task_type == 'ObjectDetection':
            self.service_type = ObjectDetection
        elif self.task_type == 'PromptedObjectDetection':
            self.service_type = PromptedObjectDetection
        elif self.task_type == 'DepthEstimation':
            self.service_type = DepthEstimation
        elif self.task_type == 'VideoClassification':
            self.service_type = VideoClassification
        # Determine the camera topic
        self.camera_topic = self.config['system']['camera_topic']
        # Determine the number of frames to collect for a video
        self.num_frames = self.config['system']['num_video_frames']
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
        if self.service_type == ObjectDetection:
            self.process_detection(img_msg)
        elif self.service_type == PromptedObjectDetection:
            self.process_prompteddetection(img_msg)
        elif self.service_type == ImageSegmentation:
            self.process_segmentation(img_msg)
        elif self.service_type == DepthEstimation:
            self.process_depthestimation(img_msg)
        elif self.service_type == VideoClassification:
            self.process_videoclassification(img_msg)


    def process_prompteddetection(self, img_msg):
        """Process image detection service requests."""
        try:
            # Convert the ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            request = PromptedObjectDetectionRequest(image=img_msg, prompt=self.prompt)
            response = self.annotator_service(request)

            # If logging level is DEBUG, display the bounding boxes
            if self.logging_level == 'DEBUG':
                self.display_bounding_boxes(cv_image, response)
        except Exception as e:
            rospy.logerr(f"Error processing detection image: {e}")

    def process_detection(self, img_msg):
        """Process image detection service requests."""
        try:
            # Convert the ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            request = ObjectDetectionRequest(image=img_msg)
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
            request = ImageSegmentationRequest(image=img_msg)
            response = self.annotator_service(request)

            # If logging level is DEBUG, display the segmentation masks
            if self.logging_level == 'DEBUG':
                self.display_segmentation_masks(cv_image, response)
        except Exception as e:
            rospy.logerr(f"Error processing segmentation image: {e}")

    def process_depthestimation(self, img_msg):
        """Process depth estimation service requests."""
        try:
            # Convert the ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            request = DepthEstimationRequest(image=img_msg)
            response = self.annotator_service(request)

            # If logging level is DEBUG, display the bounding boxes
            if self.logging_level == 'DEBUG':
                self.display_depthmap(response)
        except Exception as e:
            rospy.logerr(f"Error processing detection image: {e}")


    def process_videoclassification(self, img_msg):
        """Process depth estimation service requests."""
        try:
            # Convert the ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            self.video_frames.append(cv_image)
            if len(self.video_frames) == self.num_frames:
                ros_video_frames = [self.bridge.cv2_to_imgmsg(frame, "bgr8") for frame in self.video_frames]

                request = VideoClassificationRequest(video_frames=ros_video_frames)
                response = self.annotator_service(request)
                rospy.loginfo(f"Detected activity: {response}")
            else:
                rospy.loginfo(f"Collecting frames ({len(self.video_frames)}/{self.num_frames} frames collected)")
        except Exception as e:
            rospy.logerr(f"Error processing detection image: {e}")

# DISPLAY FUNCTIONS

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



    def display_depthmap(self, response):
        """
        Display the depth map from a DepthEstimationResponse.

        Args:
            response (DepthEstimationResponse): The response containing the depth map.

        Returns:
            None
        """
        try:
            # Extract metadata
            width = response.depth_map.width
            height = response.depth_map.height

            # Decode raw byte data if necessary
            if isinstance(response.depth_map.data, bytes):
                depth_data = np.frombuffer(response.depth_map.data, dtype=np.uint8).astype(np.float32)
            else:
                depth_data = np.array(response.depth_map.data, dtype=np.float32)

            # Check if reshaping is possible
            if depth_data.size != width * height:
                rospy.logerr(
                    f"Mismatch in depth data size ({depth_data.size}) and dimensions ({width}x{height})"
                )
                return

            # Reshape to original dimensions
            depth_data = depth_data.reshape((height, width))

            # Normalize depth values for visualization
            max_value = depth_data.max()
            if max_value <= 0:
                rospy.logerr("Invalid max value in depth data. Cannot normalize.")
                return

            formatted_depth = (depth_data * 255 / max_value).astype("uint8")

            # Display the depth map
            cv2.imshow("Depth Estimation", formatted_depth)
            rospy.loginfo("Press any key in the display window to exit.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            rospy.logerr(f"Failed to display depth map: {e}")


if __name__ == "__main__":
    try:
        annotator_client = MainAnnotatorClient()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
