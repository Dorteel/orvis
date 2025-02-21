from orvis.srv import PromptedImageClassificationResponse
import torch
import rospy
import importlib
from PIL import Image
from cv_bridge import CvBridge

class PromptedImageClassifier:
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

        self.bridge = CvBridge()  # Initialize CvBridge for image conversion

    def handle_request(self, req):
        rospy.loginfo("Handling Prompted Image Classification Task")

        # Convert ROS Image to PIL Image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(req.image, "bgr8")
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")
            return PromptedImageClassificationResponse()

        pil_image = Image.fromarray(cv_image[:, :, ::-1])

        # Prepare input and run model
        inputs = self.processor(text=req.prompts, images=pil_image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract similarity scores and compute probabilities
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).squeeze().tolist()

        # Construct response
        response = PromptedImageClassificationResponse()
        response.keys = req.prompts
        response.values = probs
        return response

    def dynamic_import(self, import_path):
        """
        Dynamically import the class from the import path string.
        For example: 'transformers.CLIPModel'
        """
        module_path, class_name = import_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)