from orvis.srv import ServiceResponseType
import torch
import rospy
import importlib
from PIL import Image
from cv_bridge import CvBridge

class YourClassName:
    def __init__(self, config):
        """
        Initialize the class with the given configuration.
        Replace YourClassName with the actual name of the classifier.
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

        self.bridge = CvBridge()  # Initializes CvBridge for image conversion

    def handle_request(self, req):
        """
        Handles incoming ROS service requests.
        Replace ServiceResponseType with the actual response type from the .srv file.
        """
        rospy.loginfo("Handling <YourTaskName> Task")

        # Convert ROS Image to PIL Image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(req.image, "bgr8")
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")
            return ServiceResponseType()

        pil_image = Image.fromarray(cv_image[:, :, ::-1])

        # Prepare input and run model
        inputs = self.processor(text=req.<YourTextField>, images=pil_image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract results and compute probabilities
        logits_per_image = outputs.<YourLogitsField>
        probs = logits_per_image.softmax(dim=1).squeeze().tolist()

        # Construct response
        response = ServiceResponseType()
        response.keys = req.<YourTextField>
        response.values = probs
        return response

    def dynamic_import(self, import_path):
        """
        Dynamically import the class from the import path string.
        Example: 'transformers.CLIPModel'
        """
        module_path, class_name = import_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
