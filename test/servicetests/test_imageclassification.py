import rospy
from sensor_msgs.msg import Image
from orvis.srv import ObjectDetection, ObjectDetectionRequest, PromptedImageClassification, PromptedImageClassificationRequest
import cv2
from cv_bridge import CvBridge
import matplotlib.pyplot as plt

class ImageClassifierTester:
    def __init__(self):
        rospy.init_node('prompted_object_detector_tester')
        self.bridge = CvBridge()
        
        # Service proxies
        rospy.wait_for_service('/annotators/ObjectDetection/yolos_tiny/detect')
        rospy.wait_for_service('/annotators/PromptedImageClassification/clip_vit_large_patch14/detect')
        
        self.object_detection_service = rospy.ServiceProxy('/annotators/ObjectDetection/yolos_tiny/detect', ObjectDetection)
        self.classification_service = rospy.ServiceProxy('/annotators/PromptedImageClassification/clip_vit_large_patch14/detect', PromptedImageClassification)
        
        self.prompts = ['salt', 'fruit', 'bottle', 'vegetable']
        self.image = None
        rospy.Subscriber('/locobot/camera/color/image_raw', Image, self.image_callback)
    
    def image_callback(self, msg):
        """Callback function to update the latest image from the ROS topic."""
        self.image = msg
    
    def run_test(self):
        """Runs the test on the latest received image."""
        if self.image is None:
            rospy.logerr("No image received from /locobot/camera/color/image_raw")
            return
        
        # Convert ROS Image to OpenCV Image
        cv_image = self.bridge.imgmsg_to_cv2(self.image, "bgr8")
        ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        
        # Step 1: Call YOLO object detection service
        detect_req = ObjectDetectionRequest()
        detect_req.image = ros_image
        detect_resp = self.object_detection_service(detect_req)
        
        self.display_bounding_boxes(cv_image, detect_resp.objects.bounding_boxes)
        
        # Step 2: Run classification on each detected bounding box
        for bbox in detect_resp.objects.bounding_boxes:
            cropped_image = cv_image[bbox.ymin:bbox.ymax, bbox.xmin:bbox.xmax]
            ros_cropped_image = self.bridge.cv2_to_imgmsg(cropped_image, encoding="bgr8")
            
            classify_req = PromptedImageClassificationRequest()
            classify_req.image = ros_cropped_image
            classify_req.prompts = self.prompts
            classify_resp = self.classification_service(classify_req)
            
            self.print_classification_results(bbox, classify_resp)
    
    def display_bounding_boxes(self, image, bounding_boxes):
        """Displays bounding boxes on the image."""
        for bbox in bounding_boxes:
            cv2.rectangle(image, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (0, 255, 0), 2)
            cv2.putText(image, bbox.Class, (bbox.xmin, bbox.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()
    
    def print_classification_results(self, bbox, classify_resp):
        """Prints classification results for a detected object."""
        print(f"Object ({bbox.Class}) classification results:")
        for key, value in zip(classify_resp.keys, classify_resp.values):
            print(f"  {key}: {value:.4f}")

if __name__ == '__main__':
    tester = ImageClassifierTester()
    rospy.sleep(2)  # Allow some time to receive an image
    tester.run_test()
