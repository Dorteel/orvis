#!/home/user/pel_ws/pel_venv/bin/python

import rospy
import yaml
import importlib
from PIL import Image  # Import this at the top
import rospkg
import torch
import io
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image as ROSImage
from orvis.srv import ImageDetection, ImageDetectionResponse
from orvis.srv import ImageMaskDetection, ImageMaskDetectionResponse  # Segmentation service
from orvis.msg import BoundingBoxes, BoundingBox  # Import the custom message types
from orvis.msg import ImageMask, ImageMasks


from transformers.models.detr.feature_extraction_detr import rgb_to_id


class ServiceManager:
    def __init__(self):
        # Initialize the list to hold services
        self.services = []

        self.bridge = CvBridge()

        # Use rospkg to get the path of the "orvis" package
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('orvis')

        # Set the paths for the main config and model configs
        main_config_path = f"{package_path}/config/main_config.yaml"
        model_config_dir = f"{package_path}/config/models"

        # Load the main config
        with open(main_config_path, 'r') as file:
            self.main_config = yaml.safe_load(file)

        self.conf_threshold = self.main_config['confidence_threshold']
        # Iterate over the annotators in the main config and load their corresponding configs
        for annotator_name in self.main_config['annotators']:
            model_config_path = f"{model_config_dir}/{annotator_name}.yaml"
            with open(model_config_path, 'r') as file:
                annotator_config = yaml.safe_load(file)
                self.create_service_from_config(annotator_config)

    def create_service_from_config(self, config):
        """Dynamically create a service from the given annotator config."""
        # Import the necessary model and processor
        model_class = self.dynamic_import(config['imports']['model_class'])
        processor_class = self.dynamic_import(config['imports']['processor_class'])

        # Load model and processor
        model = model_class.from_pretrained(config['model']['model_name'])
        processor = processor_class.from_pretrained(config['processor']['processor_name'])

        # Create a service and store it
        service_name = f"/{config['annotator']['name']}/detect"
        task_type = config['annotator']['task_type']  # Get the task type from config

        # Get labels from config
        labels = config['detection']['labels']

        # Dynamically create the service and bind the correct callback based on the task type
        if task_type == 'StandardDetectionTask':
            service = rospy.Service(service_name, ImageDetection, self.generate_callback(task_type, model, processor, labels))
        elif task_type == 'SegmentationTask':
            service = rospy.Service(service_name, ImageMaskDetection, self.generate_callback(task_type, model, processor, labels))
        self.services.append(service)
        rospy.loginfo(f"Created service: {service_name} for task: {task_type}")

    def generate_callback(self, task_type, model, processor, labels):
        """
        Generate the appropriate callback based on the task type.
        The task types can be:
        - StandardDetectionTask
        - SegmentationTask
        - DepthEstimationTask
        - HumanPoseDetectionTask
        - PromptedDetectionTask
        """
        def callback(req):
            rospy.loginfo(f"Service requested for task: {task_type}")

            if task_type == 'StandardDetectionTask':
                return self.handle_standard_detection(model, processor, req.image, labels)
            elif task_type == 'SegmentationTask':
                return self.handle_segmentation_task(model, processor, req.image, labels)
            elif task_type == 'DepthEstimationTask':
                return self.handle_depth_estimation_task(model, processor)
            elif task_type == 'HumanPoseDetectionTask':
                return self.handle_human_pose_detection_task(model, processor)
            elif task_type == 'PromptedDetectionTask':
                return self.handle_prompted_detection_task(model, processor)
            else:
                rospy.logwarn(f"Unknown task type: {task_type}")
                return 

        return callback

    def handle_standard_detection(self, model, processor, img_msg: ROSImage, labels):
        """Handle standard object detection"""

        rospy.loginfo("Handling Standard Detection Task (bounding boxes)")

        # Convert the ROS Image message to a PIL image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")
            return ImageDetectionResponse()

        # Convert the OpenCV image (BGR format) to PIL format (RGB)
        pil_image = Image.fromarray(cv_image[:, :, ::-1])  # Convert BGR to RGB

        # Process the image with the DETR processor
        inputs = processor(images=pil_image, return_tensors="pt")

        # Run the model on the processed input
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process the outputs to get the bounding boxes and labels
        target_sizes = torch.tensor([pil_image.size[::-1]])  # width, height
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=self.conf_threshold)[0]

        # Prepare the response
        bounding_boxes = BoundingBoxes()

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]

            # Create a BoundingBox message
            bbox = BoundingBox()
            bbox.Class = model.config.id2label[label.item()]
            #bbox.Class = labels[label.item()]
            
            bbox.probability = score.item()
            bbox.xmin = int(box[0])
            bbox.ymin = int(box[1])
            bbox.xmax = int(box[2])
            bbox.ymax = int(box[3])

            # Add the bounding box to the BoundingBoxes message
            bounding_boxes.bounding_boxes.append(bbox)

            rospy.loginfo(
                f"Detected {bbox.Class} with confidence "
                f"{round(bbox.probability, 3)} at location "
                f"({bbox.xmin}, {bbox.ymin}), ({bbox.xmax}, {bbox.ymax})"
            )

        # Fill in the response
        response = ImageDetectionResponse()
        response.objects = bounding_boxes

        return response

    def handle_segmentation_task(self, model, processor, img_msg: ROSImage, labels):
        """Handle segmentation task using DETR Panoptic segmentation"""

        rospy.loginfo("Handling Segmentation Task (masks)")

        # Convert the ROS Image message to a PIL image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")
            return ImageMasks()

        # Convert the OpenCV image (BGR format) to PIL format (RGB)
        pil_image = Image.fromarray(cv_image[:, :, ::-1])  # Convert BGR to RGB

        # Prepare the input for the model
        inputs = processor(images=pil_image, return_tensors="pt")

        # Run the model to get segmentation outputs
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process the results to get the segmentation masks in COCO format
        processed_sizes = torch.as_tensor(inputs["pixel_values"].shape[-2:]).unsqueeze(0)
        result = processor.post_process_panoptic(outputs, processed_sizes)[0]

        # Convert the segmentation result into a numpy array for further processing
        panoptic_seg = Image.open(io.BytesIO(result["png_string"]))
        panoptic_seg = np.array(panoptic_seg, dtype=np.uint8)
        
        # Retrieve the IDs corresponding to each mask
        panoptic_seg_id = rgb_to_id(panoptic_seg)

        # Create the response with segmentation masks
        image_masks = ImageMasks()

        # Go through each unique mask ID and create a corresponding ImageMask message
        for seg_id in np.unique(panoptic_seg_id):
            if seg_id == 0:
                continue  # Skip the background mask (ID = 0)
            
            # Create a new ImageMask message
            mask_msg = ImageMask()

            # Get the class label corresponding to this mask (if available)
            class_id = result["segments_info"][seg_id]["category_id"]
            mask_msg.Class = model.config.id2label[class_id]
            # mask_msg.Class = labels[class_id] if class_id < len(labels) else "unknown"
        
            # Check if the 'score' field exists before accessing it
            if "score" in result["segments_info"][seg_id]:
                mask_msg.probability = result["segments_info"][seg_id]["score"]
            else:
                rospy.loginfo(f"No 'score' found for segment ID: {seg_id}, setting probability to 1.0")
                mask_msg.probability = 1.0  # Default to 1.0 if score is missing

            # Convert the segmentation mask for this object into a ROS Image
            # Ensure the mask has the same dimensions as the original image
            mask = (panoptic_seg_id == seg_id).astype(np.uint8) * 255  # Binary mask for this object
            
            # Resize the mask to the original image size (if necessary)
            if mask.shape[:2] != cv_image.shape[:2]:
                mask = cv2.resize(mask, (cv_image.shape[1], cv_image.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Convert to ROS Image message, ensuring single channel (mono8)
            ros_mask = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
            mask_msg.mask = ros_mask

            # Append the mask message to the list of masks
            image_masks.masks.append(mask_msg)

            rospy.loginfo(f"Detected mask for {mask_msg.Class} with confidence {mask_msg.probability}")

        # Fill in the response
        response = ImageMasks()
        response.masks = image_masks.masks

        return response

    def handle_depth_estimation_task(self, model, processor):
        # Placeholder for the Depth Estimation logic (depthmap)
        rospy.loginfo("Handling Depth Estimation Task (depthmap)")
        return ImageDetectionResponse()

    def handle_human_pose_detection_task(self, model, processor):
        # Placeholder for the Human Pose Detection logic (pose)
        rospy.loginfo("Handling Human Pose Detection Task (pose)")
        return ImageDetectionResponse()

    def handle_prompted_detection_task(self, model, processor):
        # Placeholder for the Prompted Detection logic (bounding boxes or custom output)
        rospy.loginfo("Handling Prompted Detection Task (bounding boxes/custom)")
        return ImageDetectionResponse()

    def dynamic_import(self, import_path):
        """Dynamically imports a module or class based on the import path"""
        module_path, class_name = import_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

if __name__ == "__main__":
    rospy.init_node('service_manager')
    manager = ServiceManager()
    rospy.spin()
