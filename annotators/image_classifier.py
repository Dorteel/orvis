import torch
import rospy
import importlib
from PIL import Image
from cv_bridge import CvBridge
from orvis.srv import ImageClassificationResponse

class ImageClassifier:
    def __init__(self, config):
        self.config = config
        self.bridge = CvBridge()
        
        # Dynamically import model and processor classes
        model_class_path = config['imports']['model_class']
        self.model_class = self.dynamic_import(model_class_path)
        
        processor_class_path = config['imports']['processor_class']
        self.processor_class = self.dynamic_import(processor_class_path)
        
        # Load the model and processor
        self.model = self.model_class.from_pretrained(config['model']['model_name'])
        self.processor = self.processor_class.from_pretrained(config['processor']['processor_name'])
    
    def handle_request(self, req):
        """
        Handle an image classification request, convert ROS Image to PIL, process it, and return results.
        """
        rospy.loginfo("Handling Image Classification Request")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(req.image, "bgr8")
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")
            return ImageClassificationResponse()
        
        pil_image = Image.fromarray(cv_image[:, :, ::-1])
        results = self.process_image(pil_image)
        
        response = ImageClassificationResponse()
        response.keys = list(results.keys())
        response.values = list(results.values())
        return response
    
    def process_image(self, image):
        """
        Process a single image using the model and return class labels with probabilities.
        :param image: PIL Image instance.
        """
        rospy.loginfo("Processing image with model")
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1).tolist()[0]
        labels = self.model.config.id2label  # Retrieve label names from model config
        return {labels[i]: prob for i, prob in enumerate(probs)}
    
    def dynamic_import(self, import_path):
        """
        Dynamically import the class from the import path string.
        """
        module_path, class_name = import_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)