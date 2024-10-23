# object_detector.py
from orvis.srv import ImageDetectionResponse
from orvis.msg import BoundingBoxes, BoundingBox
import torch
import rospy
import importlib
from PIL import Image
from cv_bridge import CvBridge

class ObjectDetector:
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
        self.conf_threshold = config['detection']['confidence_threshold']

        self.bridge = CvBridge()  # Initialize CvBridge here

    def handle_request(self, req):
        rospy.loginfo("Handling Object Detection Task")

        # Convert ROS Image to PIL Image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(req.image, "bgr8")
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")
            return ImageDetectionResponse()

        pil_image = Image.fromarray(cv_image[:, :, ::-1])

        # Prepare input and run model
        inputs = self.processor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process and create bounding boxes
        target_sizes = torch.tensor([pil_image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=self.conf_threshold)[0]

        bounding_boxes = BoundingBoxes()
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            bbox = BoundingBox()
            bbox.Class = self.model.config.id2label[label.item()]
            bbox.probability = score.item()
            bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax = [int(round(i)) for i in box.tolist()]
            bounding_boxes.bounding_boxes.append(bbox)

        response = ImageDetectionResponse()
        response.objects = bounding_boxes
        return response

    def dynamic_import(self, import_path):
        """
        Dynamically import the class from the import path string.
        For example: 'transformers.DetrForObjectDetection'
        """
        module_path, class_name = import_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)