# image_segmenter.py
from orvis.srv import ImageMaskDetectionResponse
from orvis.msg import ImageMasks, ImageMask
from PIL import Image
import rospy
import torch
import numpy as np
import cv2
import importlib
from cv_bridge import CvBridge

class ImageSegmenter:
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

        self.bridge = CvBridge()  # Initialize CvBridge here
        # Additional configurations
        self.confidence_threshold = config['detection']['confidence_threshold']


    def handle_request(self, req):
        rospy.loginfo("Handling Image Segmentation Task")

        try:
            cv_image = self.bridge.imgmsg_to_cv2(req.image, "bgr8")
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")
            return ImageMaskDetectionResponse()

        pil_image = Image.fromarray(cv_image[:, :, ::-1])

        inputs = self.processor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Check if it's Segformer or DETR and process accordingly
        if hasattr(outputs, 'logits'):
            return self.process_segformer(outputs, pil_image)
        elif hasattr(outputs, 'png_string'):
            return self.process_detr_panoptic(outputs, inputs)

    def process_segformer(self, outputs, pil_image):
        logits = outputs.logits
        logits = torch.nn.functional.interpolate(logits, size=pil_image.size[::-1], mode='bilinear', align_corners=False)
        logits = torch.argmax(logits.squeeze(), dim=0).cpu().numpy()

        image_masks = ImageMasks()
        for label_id in np.unique(logits):
            if label_id == 0:
                continue
            mask_msg = ImageMask()
            mask_msg.Class = self.labels[label_id]
            mask = (logits == label_id).astype(np.uint8) * 255
            mask_msg.mask = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
            image_masks.masks.append(mask_msg)

        response = ImageMaskDetectionResponse()
        response.objects = image_masks
        return response

    def process_detr_panoptic(self, outputs, inputs):
        # Similar to Segformer, but handling DETR-specific output
        pass  # You can fill in the DETR-specific logic here

    def dynamic_import(self, import_path):
        """
        Dynamically import the class from the import path string.
        For example: 'transformers.DetrForObjectDetection'
        """
        module_path, class_name = import_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)