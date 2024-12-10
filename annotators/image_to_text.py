from orvis.srv import ImageToTextResponse
from orvis.msg import BoundingBoxes, BoundingBox
import torch
import rospy
import importlib
from PIL import Image
from cv_bridge import CvBridge


class ImageToTextConverter:
    def __init__(self, config):
        """
        Initialize the ImageToTextConverter with the given configuration.

        Args:
            config (dict): Configuration dictionary for the annotator.
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

        # Additional configurations
        self.prompt = config['generation']['prompt']
        self.max_new_tokens = config['generation']['max_new_tokens']

        self.bridge = CvBridge()  # Initialize CvBridge here

    def handle_request(self, req):
        """
        Handle the Image-to-Text request.

        Args:
            req: ROS service request containing the input image.

        Returns:
            ImageToTextResponse: ROS service response with text and entities.
        """
        rospy.loginfo("Handling Image-to-Text Task")

        # Convert ROS Image to PIL Image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(req.image, "bgr8")
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")
            return ImageToTextResponse(objects=BoundingBoxes(bounding_boxes=[]))

        pil_image = Image.fromarray(cv_image[:, :, ::-1])

        # Prepare inputs for the model
        try:
            inputs = self.processor(text=self.prompt, images=pil_image, return_tensors="pt")
        except Exception as e:
            rospy.logerr(f"Error preparing inputs: {e}")
            return ImageToTextResponse(objects=BoundingBoxes(bounding_boxes=[]))

        # Perform inference
        try:
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
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        except Exception as e:
            rospy.logerr(f"Error during inference: {e}")
            return ImageToTextResponse(objects=BoundingBoxes(bounding_boxes=[]))

        # Post-process the generated text
        try:
            processed_text, entities = self.processor.post_process_generation(generated_text)
            rospy.loginfo(f"Generated Text: {processed_text}")
        except Exception as e:
            rospy.logerr(f"Error during post-processing: {e}")
            return ImageToTextResponse(objects=BoundingBoxes(bounding_boxes=[]))

        # Create BoundingBoxes message
        bounding_boxes = BoundingBoxes()
        for entity in entities:
            description, _, boxes = entity
            for box in boxes:
                bbox = BoundingBox()
                bbox.Class = description
                bbox.xmin = int(box[0] * cv_image.shape[1])
                bbox.ymin = int(box[1] * cv_image.shape[0])
                bbox.xmax = int(box[2] * cv_image.shape[1])
                bbox.ymax = int(box[3] * cv_image.shape[0])
                bbox.probability = 1.0  # Placeholder, as the model doesn't return confidence
                bounding_boxes.bounding_boxes.append(bbox)

        # Populate the response
        response = ImageToTextResponse()
        response.objects = bounding_boxes

        return response

    def dynamic_import(self, import_path):
        """
        Dynamically import the class from the import path string.

        Args:
            import_path (str): Path to the class to import.

        Returns:
            type: The dynamically imported class.
        """
        module_path, class_name = import_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
