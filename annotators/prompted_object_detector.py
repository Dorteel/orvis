# prompted_object_detector.py
from orvis.srv import PromptedImageDetectionResponse
from orvis.msg import BoundingBoxes, BoundingBox
import torch
import rospy
import importlib
from PIL import Image
from cv_bridge import CvBridge
import numpy as np
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

class PromptedObjectDetector:
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

        # Confidence threshold
        self.conf_threshold = config['detection']['confidence_threshold']

        self.bridge = CvBridge()

    def handle_request(self, req):
        rospy.loginfo("Handling Prompted Object Detection Task with dynamic prompt")

        # Convert ROS Image to PIL Image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(req.image, "bgr8")
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")
            return PromptedImageDetectionResponse()

        pil_image = Image.fromarray(cv_image[:, :, ::-1])

        # Use the prompt from the request (dynamic text prompt)
        dynamic_prompt = req.prompt.data
        prompts = [[dynamic_prompt]]  # Ensure it's in the required nested list format

        # Prepare inputs with the dynamic text prompt and image
        inputs = self.processor(text=prompts, images=pil_image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process to get bounding boxes
        unnormalized_image = self.get_preprocessed_image(inputs.pixel_values)
        target_sizes = torch.Tensor([unnormalized_image.size[::-1]])

        results = self.processor.post_process_object_detection(
            outputs=outputs, threshold=self.conf_threshold, target_sizes=target_sizes
        )[0]

        # Create bounding boxes
        bounding_boxes = BoundingBoxes()
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            bbox = BoundingBox()
            bbox.Class = dynamic_prompt  # Use the prompt itself as the class label
            bbox.probability = score.item()
            bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax = [int(round(i)) for i in box.tolist()]
            bounding_boxes.bounding_boxes.append(bbox)

        response = PromptedImageDetectionResponse()
        response.objects = bounding_boxes
        return response

    def get_preprocessed_image(self, pixel_values):
        """Reverts the pixel values to an unnormalized state to match the image used for processing."""
        pixel_values = pixel_values.squeeze().numpy()
        unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
        unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
        unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
        return Image.fromarray(unnormalized_image)

    def dynamic_import(self, import_path):
        """
        Dynamically import the class from the import path string.
        For example: 'transformers.Owlv2ForObjectDetection'
        """
        module_path, class_name = import_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
