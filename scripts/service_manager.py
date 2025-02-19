#!/home/user/pel_ws/pel_venv/bin/python


import rospy
import yaml
import rospkg
import sys
import os
import rosservice

rospack = rospkg.RosPack()
package_path = rospack.get_path('orvis')
sys.path.append(os.path.join(package_path))

from annotators.object_detector import ObjectDetector
from annotators.image_segmenter import ImageSegmenter
from annotators.depth_estimator import DepthEstimator
from annotators.image_classifier import ImageClassifier
from annotators.pose_detector import PoseDetector
from annotators.prompted_image_classifier import PromptedImageClassifier
from annotators.video_classifier import VideoClassifier
from annotators.image_to_text import ImageToTextConverter
from annotators.prompted_object_detector import PromptedObjectDetector
from cv_bridge import CvBridge
from orvis.srv import ObjectDetection, ImageSegmentation, PromptedObjectDetection, DepthEstimation, ImageClassification, VideoClassification, ImageToText, PromptedImageClassification  # Import the necessary service types



class ServiceManager:
    def __init__(self):
        self.services = []
        self.bridge = CvBridge()

        # Load config files
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('orvis')
        main_config_path = f"{package_path}/config/orvis_config.yaml"
        model_config_dir = f"{package_path}/config/models"

        # Load the main configuration
        with open(main_config_path, 'r') as file:
            self.main_config = yaml.safe_load(file)

        # Iterate over annotators in the main config
        print("Annotators to initialize:")
        print(f"\t{self.main_config['annotators']}")
        for annotator_name in self.main_config['annotators']:
            model_config_path = f"{model_config_dir}/{annotator_name}.yaml"
            print(f"Initializing {annotator_name}")
            with open(model_config_path, 'r') as file:
                annotator_config = yaml.safe_load(file)
                self.create_service_from_config(annotator_config)

    def create_service_from_config(self, config):
        """
        Dynamically create the annotator service based on the task type.
        This method imports the appropriate class and registers a ROS service.
        """
        task_type = config['annotator']['task_type']
        service_name = f"/annotators/{task_type}/{config['annotator']['name']}/detect"
        print(f'{task_type}: {service_name}')
        if task_type == 'ImageClassification':
            annotator = ImageClassifier(config)
            service = rospy.Service(service_name, ImageClassification, annotator.handle_request)
        if task_type == 'PromptedImageClassification':
            annotator = PromptedImageClassifier(config)
            service = rospy.Service(service_name, PromptedImageClassification, annotator.handle_request)
        if task_type == 'ObjectDetection':
            annotator = ObjectDetector(config)
            service = rospy.Service(service_name, ObjectDetection, annotator.handle_request)
        elif task_type == 'ImageSegmentation':
            annotator = ImageSegmenter(config)
            service = rospy.Service(service_name, ImageSegmentation, annotator.handle_request)
        elif task_type == 'DepthEstimation':
            annotator = DepthEstimator(config)
            service = rospy.Service(service_name, DepthEstimation, annotator.handle_request)
        elif task_type == 'VideoClassification':
            annotator = VideoClassifier(config)
            service = rospy.Service(service_name, VideoClassification, annotator.handle_request)
        elif task_type == 'PromptedObjectDetection':
            annotator = PromptedObjectDetector(config)
            service = rospy.Service(service_name, PromptedObjectDetection, annotator.handle_request)
        elif task_type == 'ImageToText':
            annotator = ImageToTextConverter(config)
            service = rospy.Service(service_name, ImageToText, annotator.handle_request)
        else:
            rospy.logwarn(f"Unknown task type: {task_type}")
            return
        
        # Store the service to keep a reference
        self.services.append(service)
        rospy.loginfo(f"Created service: {service_name} for task: {task_type}")

    def run(self):
        """
        This method starts the ROS node and keeps it running.
        """
        rospy.init_node('service_manager', anonymous=True)
        rospy.loginfo("Service Manager is running...")
        rospy.spin()  # Keeps the node alive and handling service requests


if __name__ == "__main__":
    try:
        manager = ServiceManager()
        manager.run()  # Start the service manager to handle all services
    except rospy.ROSInterruptException:
        rospy.logerr("Service Manager interrupted.")
