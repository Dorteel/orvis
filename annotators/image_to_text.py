# image_to_text.py
from orvis.srv import ImageToTextResponse
import torch
import rospy
import importlib
from PIL import Image
from cv_bridge import CvBridge


class ImageToTextConverter:
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
        self.prompt = config['generation']['prompt']
        self.max_new_tokens = config['generation']['max_new_tokens']

        self.bridge = CvBridge()  # Initialize CvBridge here

    def handle_request(self, req):
        rospy.loginfo("Handling Image-to-Text Task")

        # Convert ROS Image to PIL Image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(req.image, "bgr8")
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")
            return ImageToTextResponse()

        pil_image = Image.fromarray(cv_image[:, :, ::-1])

        # Prepare input and run model
        inputs = self.processor(text=self.prompt, images=pil_image, return_tensors="pt")
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_embeds=None,
                image_embeds_position_mask=inputs["image_embeds_position_mask"],
                use_cache=True,
                max_new_tokens=self.max_new_tokens,
            )

        # Decode the generated text
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Optionally clean up and extract entities
        processed_text, entities = self.processor.post_process_generation(generated_text)

        # Populate the response
        response = ImageToTextResponse()
        response.generated_text = processed_text
        response.entities = []  # Add entity details if needed

        return response

    def dynamic_import(self, import_path):
        """
        Dynamically import the class from the import path string.
        For example: 'transformers.AutoModelForVision2Seq'
        """
        module_path, class_name = import_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
