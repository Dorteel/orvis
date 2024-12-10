# image_segmenter.py
from orvis.srv import ImageSegmentationResponse
from orvis.msg import ImageMasks, ImageMask
from PIL import Image
import rospy
import torch
import numpy as np
import cv2
import importlib
from cv_bridge import CvBridge
from transformers.models.detr.feature_extraction_detr import rgb_to_id
import io


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

        self.annotator_type = config['annotator']['type']

        self.bridge = CvBridge()  # Initialize CvBridge here
        # Additional configurations
        self.confidence_threshold = config['detection']['confidence_threshold']


    def handle_request(self, req):
        rospy.loginfo("Handling Image Segmentation Task")

        try:
            cv_image = self.bridge.imgmsg_to_cv2(req.image, "bgr8")
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")
            return ImageSegmentationResponse()

        pil_image = Image.fromarray(cv_image[:, :, ::-1])

        inputs = self.processor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Check if it's Segformer or DETR and process accordingly
        if self.annotator_type == 'Segformer':
            return self.process_segformer(outputs, pil_image)
        elif self.annotator_type == 'DETR_Panoptic':
            # Pass the correct arguments here
            return self.process_detr_panoptic(outputs, inputs, pil_image)

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

        response = ImageSegmentationResponse()
        response.objects = image_masks
        return response

    def process_detr_panoptic(self, outputs, inputs, pil_image):
        """Process the outputs from the DETR panoptic model."""
        
        # Post-process the outputs to get the panoptic segmentation in COCO format
        processed_sizes = torch.as_tensor(inputs["pixel_values"].shape[-2:]).unsqueeze(0)
        
        # DETR-specific post-processing using self.processor
        panoptic_result = self.processor.post_process_panoptic(outputs, processed_sizes)[0]

        # The segmentation result is stored in a special-format PNG
        panoptic_seg = Image.open(io.BytesIO(panoptic_result["png_string"]))
        panoptic_seg = np.array(panoptic_seg, dtype=np.uint8)
        
        # Retrieve the IDs corresponding to each mask
        panoptic_seg_id = rgb_to_id(panoptic_seg)

        # Prepare the response message
        image_masks = ImageMasks()

        # Loop through each segment and process the masks
        for seg_info in panoptic_result["segments_info"]:
            seg_id = seg_info["id"]  # ID for this particular segment
            category_id = seg_info["category_id"]
            
            # Create a binary mask for this segment
            mask = (panoptic_seg_id == seg_id).astype(np.uint8) * 255  # Mask of 0s and 255s

            # Resize the mask to match the original image size if needed
            mask_resized = cv2.resize(mask, (pil_image.width, pil_image.height), interpolation=cv2.INTER_NEAREST)
            
            # Create an ImageMask message
            mask_msg = ImageMask()
            mask_msg.Class = self.model.config.id2label[category_id]
            mask_msg.mask = self.bridge.cv2_to_imgmsg(mask_resized, encoding="mono8")

            # Add the mask message to the ImageMasks
            image_masks.masks.append(mask_msg)

        # Return the response
        response = ImageSegmentationResponse()
        response.objects = image_masks  # Assign the ImageMasks object to the `objects` attribute
        return response

    def dynamic_import(self, import_path):
        """
        Dynamically import the class from the import path string.
        For example: 'transformers.DetrForObjectDetection'
        """
        module_path, class_name = import_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)