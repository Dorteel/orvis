from orvis.srv import DepthEstimationResponse
from sensor_msgs.msg import Image
import rospy
import importlib
from cv_bridge import CvBridge
from PIL import Image as PILImage
import numpy as np
import torch

class DepthEstimator:
    def __init__(self, config):
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

        # Additional configurations
        self.interpolation_mode = config['detection']['interpolation_mode']
        self.align_corners = config['detection']['align_corners']

        self.bridge = CvBridge()  # Initialize CvBridge for ROS image handling

    def handle_request(self, req):
        rospy.loginfo("Handling Depth Estimation Task")

        # Convert ROS Image to PIL Image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(req.image, "bgr8")
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")
            return DepthEstimationResponse()

        pil_image = PILImage.fromarray(cv_image[:, :, ::-1])

        # Prepare input and run the model
        inputs = self.processor(images=pil_image, return_tensors="pt")

        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth

            # Interpolate to the original size
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=pil_image.size[::-1],
                mode=self.interpolation_mode,
                align_corners=self.align_corners,
            )

            # Format depth map for visualization
            depth_map = prediction.squeeze().cpu().numpy()
            formatted_depth = (depth_map * 255 / np.max(depth_map)).astype("uint8")
            
        except Exception as e:
            rospy.logerr(f"Depth estimation failed: {e}")
            return DepthEstimationResponse()

        # Create and populate the response
        response = DepthEstimationResponse()
        response.depth_map.data = formatted_depth.flatten().tolist()  # Convert to list for ROS message compatibility
        response.depth_map.width = pil_image.width
        response.depth_map.height = pil_image.height
        rospy.loginfo("Depth Estimation completed successfully")
        return response

    def dynamic_import(self, import_path):
        """
        Dynamically import the class from the import path string.
        For example: 'transformers.DPTForDepthEstimation'
        """
        module_path, class_name = import_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
