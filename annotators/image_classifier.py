import os
import torch
import importlib
import matplotlib.pyplot as plt
from PIL import Image
from cv_bridge import CvBridge
import rospy

class ImageClassifier:
    def __init__(self, config):
        self.config = config

        # Dynamically import model and processor classes
        model_class_path = config['imports']['model_class']
        self.model_class = self.dynamic_import(model_class_path)
        
        processor_class_path = config['imports']['processor_class']
        self.processor_class = self.dynamic_import(processor_class_path)

        # Load the model and processor
        self.model = self.model_class.from_pretrained(config['model']['model_name'])
        self.processor = self.processor_class.from_pretrained(config['processor']['processor_name'])
        
        self.bridge = CvBridge()
    
    def process_image(self, image, text_prompts):
        """
        Process a single image using the model.
        
        :param image: PIL Image instance.
        :param text_prompts: List of text prompts to compare against images.
        """
        rospy.loginfo("Processing image with model")
        inputs = self.processor(text=text_prompts, images=image, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).tolist()[0]
        
        return {text: prob for text, prob in zip(text_prompts, probs)}
    
    def handle_request(self, req):
        """
        Handle an image request, converting ROS Image to PIL, processing, and returning results.
        """
        rospy.loginfo("Handling Image Processing Request")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(req.image, "bgr8")
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")
            return {}
        
        pil_image = Image.fromarray(cv_image[:, :, ::-1])
        text_prompts = req.prompt.data.split(",")  # Assuming comma-separated prompts
        return self.process_image(pil_image, text_prompts)
    
    def dynamic_import(self, import_path):
        """
        Dynamically import the class from the import path string.
        """
        module_path, class_name = import_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

# Example usage
if __name__ == "__main__":
    config = {
        "imports": {
            "model_class": "transformers.CLIPModel",
            "processor_class": "transformers.CLIPProcessor"
        },
        "model": {
            "model_name": "openai/clip-vit-large-patch14"
        },
        "processor": {
            "processor_name": "openai/clip-vit-large-patch14"
        }
    }
    processor = GenericVisionProcessor(config)
    directory = "/home/user/pel_ws/src/orvis/obs_graphs/run_20250205160423"
    prompts = ["a photo of a cat", "a photo of a dog"]
    
    print("\nFinal Output:")
    for image_name in os.listdir(directory):
        if image_name.lower().endswith(".jpg"):
            image_path = os.path.join(directory, image_name)
            try:
                image = Image.open(image_path).convert("RGB")
                results = processor.process_image(image, prompts)
                print(f"Results for {image_name}:")
                for text, prob in results.items():
                    print(f"  {text}: {prob:.4f}")
            except Exception as e:
                print(f"Error processing {image_name}: {e}")