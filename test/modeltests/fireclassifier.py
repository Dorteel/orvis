import rospy
import torch
import cv2
import numpy as np
from sensor_msgs.msg import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification
from cv_bridge import CvBridge
import matplotlib.pyplot as plt

class FireDetectionTester:
    def __init__(self):
        rospy.init_node('fire_detection_tester')
        self.bridge = CvBridge()
        
        # Load model and processor
        self.model_name = "prithivMLmods/Fire-Detection-Engine"
        self.processor = ViTFeatureExtractor.from_pretrained(self.model_name)
        self.model = ViTForImageClassification.from_pretrained(self.model_name)
        
        self.image = None
        rospy.Subscriber('/locobot/camera/color/image_raw', Image, self.image_callback)
    
    def image_callback(self, msg):
        """Callback function to update the latest image from the ROS topic."""
        self.image = msg
    
    def run_test(self):
        """Runs fire detection on the latest received image."""
        if self.image is None:
            rospy.logerr("No image received from /locobot/camera/color/image_raw")
            return
        
        # Convert ROS Image to OpenCV Image
        cv_image = self.bridge.imgmsg_to_cv2(self.image, "bgr8")
        
        # Preprocess image for model input
        pil_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=pil_image, return_tensors="pt")
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get label
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=-1).item()
        labels = ["Fire Needed Action", "Normal Conditions", "Smoky Environment"]
        predicted_text = labels[predicted_label]
        
        # Display result
        self.display_image(cv_image, predicted_text)
    
    def display_image(self, image, label):
        """Displays the image with classification label."""
        cv2.putText(image, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

if __name__ == '__main__':
    tester = FireDetectionTester()
    rospy.sleep(2)  # Allow some time to receive an image
    tester.run_test()
