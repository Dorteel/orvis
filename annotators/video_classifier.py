from orvis.srv import VideoClassificationResponse
import cv2
import torch
import numpy as np
import rospy
import importlib
from cv_bridge import CvBridge
from std_msgs.msg import String


class VideoClassifier:
    def __init__(self, config):
        """
        Initialize the VideoClassifier with the given configuration.

        Args:
            config (dict): Configuration for the video classifier.
        """
        self.config = config

        # Dynamically import the model class
        model_class_path = config['imports']['model_class']
        self.model_class = self.dynamic_import(model_class_path)

        # Dynamically import the processor class
        processor_class_path = config['imports']['processor_class']
        self.processor_class = self.dynamic_import(processor_class_path)

        # Load the model and processor
        self.model = self.model_class.from_pretrained(config['model']['model_name'])
        self.processor = self.processor_class.from_pretrained(config['processor']['processor_name'])

        # Configuration parameters
        self.num_frames = config['classification']['num_frames']
        self.frame_height = config['classification']['frame_height']
        self.frame_width = config['classification']['frame_width']

        self.bridge = CvBridge()  # Initialize CvBridge for ROS message handling

    def handle_request(self, req):
        """
        Handle the video classification request.

        Args:
            req: The service request containing the video frames.

        Returns:
            VideoClassificationResponse: The response with classification results.
        """
        rospy.loginfo("Handling Video Classification Request")

        # Convert sensor_msgs/Image to numpy.ndarray
        video_frames = []
        try:
            for ros_image in req.video_frames:
                cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")  # Convert ROS Image to OpenCV format
                rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                video_frames.append(rgb_image)
        except Exception as e:
            rospy.logerr(f"Error converting ROS image to numpy.ndarray: {e}")
            response = VideoClassificationResponse()
            response.activity = String(data="Error converting ROS image")
            return response

        # Pre-process video frames for the model
        try:
            inputs = self.processor(images=video_frames, return_tensors="pt")
        except Exception as e:
            rospy.logerr(f"Error preprocessing video frames: {e}")
            response = VideoClassificationResponse()
            response.activity = String(data="Error preprocessing video frames")
            return response

        # Perform inference
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            activity = self.model.config.id2label[predicted_class_idx]
            rospy.loginfo(f"Predicted class: {activity}")
        except Exception as e:
            rospy.logerr(f"Error during inference: {e}")
            response = VideoClassificationResponse()
            response.activity = String(data="Error during inference")
            return response

        # Return the response
        response = VideoClassificationResponse()
        response.activity = String(data=activity)
        return response

    def dynamic_import(self, import_path):
        """
        Dynamically import a class from a given import path.

        Args:
            import_path (str): Path to the class to import.

        Returns:
            type: The imported class.
        """
        module_path, class_name = import_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
