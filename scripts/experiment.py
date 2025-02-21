#!/home/user/pel_ws/pel_venv/bin/python
import rospy
import rospkg
import actionlib
import tf2_ros
import uuid
import yaml
import cv2
import random
import os
import numpy as np
import time
from scipy.spatial import distance

from datetime import datetime
from collections import deque

from orvis.srv import Get3DCoordinates, Get3DCoordinatesRequest
from orvis.srv import ObjectDetection, ObjectDetectionRequest  # Detection service
from orvis.srv import ImageSegmentation, ImageSegmentationRequest  # Segmentation service
from orvis.msg import ImageMasks, ImageMask  # Import the segmentation message types
from orvis.msg import PickPlaceAction, PickPlaceGoal
from orvis.srv import PromptedObjectDetection, PromptedObjectDetectionRequest  # Detection service
from orvis.srv import DepthEstimation, DepthEstimationRequest  # Import the necessary service types
from orvis.srv import VideoClassification, VideoClassificationRequest  # Detection service
from orvis.srv import ImageToText, ImageToTextRequest  # Detection service
from orvis.srv import AssignColour, AssignColourRequest
from orvis.srv import PromptedImageClassification, PromptedImageClassificationRequest  # Detection service

from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import TransformStamped
import tf2_geometry_msgs  # Ensure compatibility with tf2

from owlready2 import get_ontology, default_world, sync_reasoner_pellet

from cv_bridge import CvBridge

class TaskSelector:
    def __init__(self):
        # Initialize the node
        rospy.init_node('task_selector')

        # Load configuration and determine service type
        self.bridge = CvBridge()
        self.load_config()
        self.prompt = String()
        self.prompt.data = "Person"

        self.last_image = None  # To store the latest image received

        # Other initializations
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.obs_graph_dir = os.path.join(os.path.dirname(self.script_dir), "obs_graphs")
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.run_id = 'orvis_orka_' + self.timestamp + '.owl'
        self.run_folder = os.path.join(self.obs_graph_dir, f"run_{self.timestamp}")
        os.makedirs(self.run_folder, exist_ok=True)
        self.save_dir = os.path.join(self.run_folder, self.run_id)
        self.service_name = ''

        # Initialize the ROS service
        self.video_frames = deque(maxlen=self.num_frames)

        # Load the ontology
        self.orka = get_ontology(self.orka_path).load()
        self.sosa = self.orka.get_namespace("http://www.w3.org/ns/sosa/")
        self.oboe = self.orka.get_namespace("http://ecoinformatics.org/oboe/oboe.1.2/oboe-core.owl#")
        self.ssn = self.orka.get_namespace("http://www.w3.org/ns/ssn/")

        # Colour calculation service
        self.assign_color_service = rospy.ServiceProxy('/assign_color', AssignColour)

        # Subscribe to the appropriate image topic
        rospy.Subscriber(self.camera_topic, Image, self.image_callback)
        rospy.loginfo("Experiment initialized.")

    def load_config(self):
        """Load main configuration and determine service parameters."""
        # Use rospkg to get the path of the "orvis" package
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('orvis')
        main_config_path = f"{package_path}/config/orvis_config.yaml"

        # Load the main config file
        with open(main_config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # Determine the camera topic
        self.camera_topic = self.config['system']['camera_topic']
        # Determine ORKA path
        self.orka_path = self.config['system']['orka_path']
        # Determine the number of frames to collect for a video
        self.num_frames = self.config['system']['num_video_frames']
        # Determine logging level
        self.logging_level = self.config['system']['logging_level']
        # Set parent frame
        self.parent_frame = self.config['system']['parent_frame']

    def image_callback(self, img_msg):
        """Store the last received image."""
        self.last_image = img_msg

    #==============================
    # Creating an observation graph
    #------------------------------

    def create_obs_graph(self, result, input_img):
        """
        Creates an observation graph
        """
        timestamp_obs = datetime.now().strftime("%Y%m%d%H%M%S")
        observation_id = 'obs_' + timestamp_obs  # Format: YYYYMMDDHHMMSS
        obs_instance = self.oboe.Observation(observation_id) # Create an instance of Observation
        rospy.loginfo(f"Creating observation graph for {observation_id}")
        procedure_instance = self.sosa.Procedure(self.service_name)
        
        if self.service_type == ObjectDetection or self.service_type == PromptedObjectDetection or self.service_type == ImageToText:
            for bbox in result.objects.bounding_boxes:
                rospy.loginfo(f'... Creating observation for {str(self.service_type)} result: {bbox.Class}')
                coordinates = self.create_3d_coordinates(bbox)
                rospy.loginfo(f"... ... the calculated coordinates are {coordinates}")
                if not coordinates:
                    rospy.loginfo(f"... ... Incorrect coordinates recieved for {bbox.Class}, entity skipped.")
                    continue
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S") + str(random.randint(1000, 9999))
                entity_name = 'ent_' + timestamp + '_' + self.service_name
                image_path = os.path.join(self.run_folder, f'{entity_name}_{bbox.Class}.png')
                # Creating instances
                try:
                    ent_instance = self.orka[bbox.Class.capitalize()](entity_name)
                except Exception as e:
                    rospy.loginfo(f"... ... Class \"{bbox.Class.capitalize()}\" not found. Defaulting to PhysicalEntity")
                    ent_instance = self.orka['PhysicalEntity'](entity_name)
                char_instance = self.orka.Location('loc_' + timestamp)
                mes_instance = self.oboe.Measurement('mes_' + timestamp)
                result_instance = self.sosa.Result('res_' + timestamp)
                
                cropped_img = input_img[bbox.ymin:bbox.ymax, bbox.xmin:bbox.xmax]
                cv2.imwrite(image_path, cropped_img)
                rospy.loginfo(f"... ... Cropped image saved at: {image_path}")
                # Adding properties
                obs_instance.hasMeasurement.append(mes_instance)
                mes_instance.hasResult.append(result_instance)
                mes_instance.ofCharacteristic = char_instance
                char_instance.characteristicFor = ent_instance
                obs_instance.ofEntity = ent_instance
                mes_instance.usedProcedure.append(procedure_instance)
                # Adding data properties
                result_instance.hasValue.append(image_path)
                result_instance.hasProbability.append(bbox.probability)
                char_instance.hasValue.append(str(coordinates))

                # Adding color to KG (TODO: Could be a separate function)
                # Color calculation
                assigned_color = self.process_color(cropped_img)
                color_class = self.find_color_class(assigned_color)
                
                mes_instance_color = self.oboe.Measurement('mes_color_' + timestamp)
                obs_instance.hasMeasurement.append(mes_instance_color)
                color_class_instance = None
                for cls in self.orka.classes():
                    if str(cls) == color_class or cls.iri == color_class:  # Match name or full IRI
                        color_class_instance = cls
                        break
                if color_class_instance:
                    char_color_instance = color_class_instance('color_' + timestamp + '_' + str(color_class_instance).split('.')[-1])
                else:
                    rospy.logwarn(f"Color class {color_class} not found in ontology. Defaulting to orka:Color.")
                    char_color_instance = self.orka.Color('color_' + timestamp)  # Default to generic orka:Color
                char_color_instance.characteristicFor = ent_instance
                result_color_instance = self.sosa.Result('res_color' + timestamp)
                mes_instance_color.ofCharacteristic = char_color_instance
                mes_instance_color.hasResult.append(result_color_instance)
                result_color_instance.hasValue.append(str(assigned_color))

        elif self.service_type == ImageSegmentation:
            for imagemask in result.objects.masks:
                rospy.loginfo(f'Creating observation for {imagemask.Class}')
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S") + str(random.randint(1000, 9999))
                entity_name = 'ent_' + timestamp + '_' + self.service_name
                image_path = os.path.join(self.run_folder, f'{entity_name}_{imagemask.Class}.png')
                coordinates = self.create_3d_coordinates(imagemask)
                # Creating instances
                try:
                    ent_instance = self.orka[imagemask.Class.capitalize()](entity_name)
                except Exception as e:
                    rospy.loginfo(f"Class \"{imagemask.Class.capitalize()}\" not found. Defaulting to PhysicalEntity")
                    ent_instance = self.orka['PhysicalEntity'](entity_name)
                char_instance = self.orka.Location('loc_' + timestamp)
                mes_instance = self.oboe.Measurement('mes_' + timestamp)
                result_instance = self.sosa.Result('res_' + timestamp)
                cv_image = self.bridge.imgmsg_to_cv2(imagemask.mask, "bgr8")
                cv_image = cv2.resize(cv_image, (input_img.shape[1], input_img.shape[0])) 

                # Convert the mask to grayscale to use as an actual mask
                mask_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                # Ensure mask is binary (some segmentation masks might have soft edges)
                _, mask_binary = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)
                # Apply the mask to keep the colors
                masked_img = cv2.bitwise_and(input_img, input_img, mask=mask_binary.astype(np.uint8))
                cv2.imwrite(image_path, masked_img)

                # Adding properties
                obs_instance.hasMeasurement.append(mes_instance)
                mes_instance.hasResult.append(result_instance)
                mes_instance.ofCharacteristic = char_instance
                char_instance.characteristicFor = ent_instance
                obs_instance.ofEntity = ent_instance
                mes_instance.usedProcedure.append(procedure_instance)

                # Adding data properties
                result_instance.hasValue.append(image_path)
                result_instance.hasProbability.append(imagemask.probability)
                if coordinates:
                    char_instance.hasValue.append(str(coordinates))

                # Adding color to KG (TODO: Could be a separate function)
                # Color calculation
                assigned_color = self.process_color(masked_img, input_type="mask")
                color_class = self.find_color_class(assigned_color)
                
                mes_instance_color = self.oboe.Measurement('mes_color_' + timestamp)
                obs_instance.hasMeasurement.append(mes_instance_color)
                color_class_instance = None
                for cls in self.orka.classes():
                    if str(cls) == color_class or cls.iri == color_class:  # Match name or full IRI
                        color_class_instance = cls
                        break

                if color_class_instance:
                    char_color_instance = color_class_instance('color_' + timestamp + '_' + str(color_class_instance).split('.')[-1])
                else:
                    rospy.logwarn(f"Color class {color_class} not found in ontology. Defaulting to orka:Color.")
                    char_color_instance = self.orka.Color('color_' + timestamp)  # Default to generic orka:Color
                char_color_instance.characteristicFor = ent_instance
                result_color_instance = self.sosa.Result('res_color' + timestamp)
                mes_instance_color.ofCharacteristic = char_color_instance
                mes_instance_color.hasResult.append(result_color_instance)
                result_color_instance.hasValue.append(str(assigned_color))

        elif self.service_type == DepthEstimation:
            rospy.loginfo(f'Creating observation for Depth Map')
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

            depth_map = result.depth_map.data
            # Creating instances
            ent_instance = self.orka['PhysicalEntity']('ent_' + timestamp)
            char_instance = self.orka.Location('loc_' + timestamp)
            mes_instance = self.oboe.Measurement('mes_' + timestamp)
            result_instance = self.sosa.Result('res_' + timestamp)

            # Adding properties
            obs_instance.hasMeasurement.append(mes_instance)
            mes_instance.hasResult.append(result_instance)
            mes_instance.ofCharacteristic = char_instance
            char_instance.characteristicFor = ent_instance
            obs_instance.ofEntity = ent_instance
            mes_instance.usedProcedure.append(procedure_instance)
            # Adding data properties
            result_instance.hasValue.append(depth_map)
            char_instance.hasValue.append(depth_map)
            
        elif self.service_type == VideoClassification:
            rospy.loginfo(f'Creating observation for {result.activity}')
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

            # Creating instances
            ent_instance = self.orka['PhysicalEntity']('ent_' + timestamp)
            char_instance = self.orka.ActivityType('activity_' + timestamp)
            mes_instance = self.oboe.Measurement('mes_' + timestamp)
            result_instance = self.sosa.Result('res_' + timestamp)

            # Adding properties
            obs_instance.hasMeasurement.append(mes_instance)
            mes_instance.hasResult.append(result_instance)
            mes_instance.ofCharacteristic = char_instance
            char_instance.characteristicFor = ent_instance
            obs_instance.ofEntity = ent_instance
            mes_instance.usedProcedure.append(procedure_instance)
            # Adding data properties
            result_instance.hasValue.append(str(result.activity.data))
            char_instance.hasValue.append(str(result.activity.data))
        default_world.save()
        self.orka.save(self.save_dir)



    def find_color_class(self, assigned_color):
        """
        Looks at the assigned colour in ORKA and finds the closest colour class
        """
        # Get the classes from ORKA
        sparql_query_color = """
        PREFIX orka: <https://w3id.org/def/orka#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        SELECT DISTINCT ?color ?colorvalue
        WHERE {
        ?color rdfs:subClassOf orka:Color .
        ?color rdfs:subClassOf [ owl:hasValue ?colorvalue ; owl:onProperty orka:hasRGBvalue ] .
        }
        """
        try:
            results = list(default_world.sparql(sparql_query_color))
            color_map = {str(color): str(colorvalue) for color, colorvalue in results}

            # Convert hex to RGB
            assigned_rgb = np.array([int(assigned_color[i:i+2], 16) for i in (0, 2, 4)])
            min_dist = float('inf')
            closest_color = assigned_color

            for color, hex_value in color_map.items():
                color_rgb = np.array([int(hex_value[i:i+2], 16) for i in (0, 2, 4)])
                dist = distance.euclidean(assigned_rgb, color_rgb)
                if dist < min_dist:
                    min_dist = dist
                    closest_color = color
            return closest_color
        except Exception as e:
            rospy.logerr(f"Error finding closest ORKA color: {e}")
            return assigned_color  

    def call_service(self, service_to_call, prompt=None):
        """
        Call the specified service using the last received image.

        :param service_to_call: A string representing the service to call
                                (e.g., 'ObjectDetection', 'ImageSegmentation', etc.).
        """
        # Wait until at least one image is received
        while not rospy.is_shutdown() and self.last_image is None:
            rospy.loginfo(f"Waiting for an image on topic: {self.camera_topic}")
            rospy.sleep(0.1)  # Sleep for a short duration to avoid busy-waiting

        self.prompt.data = prompt
        task_type = service_to_call.split('/')[2]
        self.service_name = service_to_call.split('/')[3] 

        if task_type == 'ImageSegmentation':
            self.service_type = ImageSegmentation 
        elif task_type == 'ObjectDetection':
            self.service_type = ObjectDetection
        elif task_type == 'PromptedObjectDetection':
            self.service_type = PromptedObjectDetection
        elif task_type == 'DepthEstimation':
            self.service_type = DepthEstimation
        elif task_type == 'VideoClassification':
            self.service_type = VideoClassification
        elif task_type == 'ImageToText':
            self.service_type = ImageToText
        elif task_type == 'PromptedImageClassification':
            self.service_type = PromptedImageClassification

        else:
            raise NameError("Service type not recognized. Check the name of the services.")


        rospy.wait_for_service(service_to_call)
        self.annotator_service = rospy.ServiceProxy(service_to_call, self.service_type)
        rospy.loginfo(f"Service {service_to_call} connected.")

        try:
            # Dispatch the request to the appropriate service processing method
            if task_type == 'ObjectDetection':
                self.process_detection(self.last_image)
            if task_type == 'PromptedImageClassification':
                self.process_promptedclassification(self.last_image)
            elif task_type == 'PromptedObjectDetection':
                self.process_prompteddetection(self.last_image)
            elif task_type == 'ImageSegmentation':
                self.process_segmentation(self.last_image)
            elif task_type == 'DepthEstimation':
                self.process_depthestimation(self.last_image)
            elif task_type == 'VideoClassification':
                self.process_videoclassification(self.last_image)
            elif task_type == 'ImageToText':
                self.process_image_to_text(self.last_image)
            else:
                rospy.logerr(f"Unknown service type: {service_to_call}. Cannot process the request.")
        except Exception as e:
            rospy.logerr(f"Error calling service {service_to_call}: {e}")


    #===========================
    # IMAGE PROCESSING FUNCTIONS
    #---------------------------

    def process_color(self, image, input_type="bounding_box"):
        """Calls the assign_color service with the provided image and input type."""
        try:
            # Convert OpenCV image to ROS Image message
            image_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")

            # Create a request
            request = AssignColourRequest()
            request.image = image_msg
            request.input_type = input_type  # Can be "mask" or "bounding_box"

            # Call the service
            response = self.assign_color_service(request)

            # Handle the response
            if response.success:
                rospy.loginfo(f"Assigned color: {response.hex_color}")
            else:
                rospy.logwarn(f"Color assignment failed: {response.message}")
            return response.hex_color

        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return None

    def process_promptedclassification(self, img_msg):
        """Process image detection service requests."""
        try:
            # Convert the ROS Image message to OpenCV format
            # Determine the image encoding and handle accordingly
            if img_msg.encoding == "16UC1":
                # Convert the ROS Image message to OpenCV format for depth images
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, "16UC1")
            else:
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            request = PromptedImageClassificationRequest(image=img_msg, prompt=self.prompt)
            response = self.annotator_service(request)

            # If logging level is DEBUG, display the bounding boxes
            if self.logging_level == 'DEBUG': self.display_bounding_boxes(cv_image, response)

            self.create_obs_graph(response, cv_image)
            return response

        except Exception as e:
            rospy.logerr(f"Error processing detection image: {e}")

    def process_prompteddetection(self, img_msg):
        """Process image detection service requests."""
        try:
            # Convert the ROS Image message to OpenCV format
            # Determine the image encoding and handle accordingly
            if img_msg.encoding == "16UC1":
                # Convert the ROS Image message to OpenCV format for depth images
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, "16UC1")
            else:
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            request = PromptedObjectDetectionRequest(image=img_msg, prompt=self.prompt)
            response = self.annotator_service(request)

            # If logging level is DEBUG, display the bounding boxes
            if self.logging_level == 'DEBUG': self.display_bounding_boxes(cv_image, response)

            self.create_obs_graph(response, cv_image)
            return response

        except Exception as e:
            rospy.logerr(f"Error processing detection image: {e}")

    def process_image_to_text(self, img_msg):
        """Process image detection service requests."""
        try:
            # Convert the ROS Image message to OpenCV format
            # Determine the image encoding and handle accordingly
            if img_msg.encoding == "16UC1":
                # Convert the ROS Image message to OpenCV format for depth images
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, "16UC1")
            else:
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            request = ImageToTextRequest(image=img_msg)
            response = self.annotator_service(request)

            # If logging level is DEBUG, display the bounding boxes
            if self.logging_level == 'DEBUG': self.display_bounding_boxes(cv_image, response)
            
            self.create_obs_graph(response, cv_image)
            return response
        except Exception as e:
            rospy.logerr(f"Error processing detection image: {e}")

    def process_detection(self, img_msg):
        """Process image detection service requests."""
        try:
            # Convert the ROS Image message to OpenCV format
            # Determine the image encoding and handle accordingly
            if img_msg.encoding == "16UC1":
                # Convert the ROS Image message to OpenCV format for depth images
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, "16UC1")
            else:
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            request = ObjectDetectionRequest(image=img_msg)
            response = self.annotator_service(request)

            # If logging level is DEBUG, display the bounding boxes
            if self.logging_level == 'DEBUG': self.display_bounding_boxes(cv_image, response)

            self.create_obs_graph(response, cv_image)
            return response
        
        except Exception as e:
            rospy.logerr(f"Error processing detection image: {e}")

    def process_segmentation(self, img_msg):
        """Process segmentation service requests."""
        try:
            # Convert the ROS Image message to OpenCV format
            # Determine the image encoding and handle accordingly
            if img_msg.encoding == "16UC1":
                # Convert the ROS Image message to OpenCV format for depth images
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, "16UC1")
            else:
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            request = ImageSegmentationRequest(image=img_msg)
            response = self.annotator_service(request)

            # If logging level is DEBUG, display the segmentation masks
            if self.logging_level == 'DEBUG': self.display_segmentation_masks(cv_image, response)

            self.create_obs_graph(response, cv_image)
            return response
        
        except Exception as e:
            rospy.logerr(f"Error processing segmentation image: {e}")

    def process_depthestimation(self, img_msg):
        """Process depth estimation service requests."""
        try:
            # Convert the ROS Image message to OpenCV format
            # Determine the image encoding and handle accordingly
            if img_msg.encoding == "16UC1":
                # Convert the ROS Image message to OpenCV format for depth images
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, "16UC1")
            else:
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            request = DepthEstimationRequest(image=img_msg)
            response = self.annotator_service(request)

            # If logging level is DEBUG, display the bounding boxes
            if self.logging_level == 'DEBUG': self.display_depthmap(response)
            
            self.create_obs_graph(response, cv_image)
            return response
        except Exception as e:
            rospy.logerr(f"Error processing detection image: {e}")

    def process_videoclassification(self, img_msg):
        """Process depth estimation service requests."""
        try:
            # Convert the ROS Image message to OpenCV format
            # Determine the image encoding and handle accordingly
            if img_msg.encoding == "16UC1":
                # Convert the ROS Image message to OpenCV format for depth images
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, "16UC1")
            else:
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            self.video_frames.append(cv_image)
            if len(self.video_frames) == self.num_frames:
                ros_video_frames = [self.bridge.cv2_to_imgmsg(frame, "bgr8") for frame in self.video_frames]

                request = VideoClassificationRequest(video_frames=ros_video_frames)
                response = self.annotator_service(request)
                rospy.loginfo(f"Detected activity: {response}")
                self.create_obs_graph(response, cv_image)
                return response
            
            else:
                rospy.loginfo(f"Collecting frames ({len(self.video_frames)}/{self.num_frames} frames collected)")
        except Exception as e:
            rospy.logerr(f"Error processing detection image: {e}")

    def get_obs_graph(self):
        """
        Fetches and loads the most recently modified observation graph (.owl file) 
        from the current run folder using owlready2.
        Returns the loaded ontology or None if no .owl file exists.
        """
        rospy.loginfo("... Fetching observation graph...")

        # Get all .owl files in the current run folder
        owl_files = [os.path.join(self.run_folder, f) for f in os.listdir(self.run_folder) if f.endswith(".owl")]

        # If no .owl files are found, return None
        if not owl_files:
            rospy.logwarn(f"No .owl files found in {self.run_folder}.")
            return None

        # Find the most recently modified .owl file
        latest_obs_graph_path = max(owl_files, key=os.path.getmtime)

        # Load the ontology using owlready2
        try:
            ontology = default_world.get_ontology(latest_obs_graph_path).load()
            rospy.loginfo(f"Ontology successfully loaded from {latest_obs_graph_path}.")
            return ontology
        except Exception as e:
            rospy.logerr(f"Failed to load ontology: {e}")
            return None


    #========================
    # 3D CALULATION FUNCTIONS
    #------------------------
    def create_3d_coordinates(self, response):
        """
        Creates 3D coordinates from a boundingbox or image
        """
        # Calculate middle depending on bbox vs mask
        if hasattr(response, 'mask'):
            bridge = CvBridge()
            try:
                # Convert the mask to a numpy array
                mask_array = bridge.imgmsg_to_cv2(response.mask, desired_encoding="mono8")
                # Find the indices of non-zero pixels in the mask
                non_zero_indices = np.argwhere(mask_array > 0)

                if non_zero_indices.size == 0:
                    rospy.logerr("Mask is empty or does not contain any non-zero pixels.")
                    return None

                # Calculate the center of the non-zero pixels
                pixel_y, pixel_x = np.mean(non_zero_indices, axis=0).astype(int)
            except Exception as e:
                rospy.logerr(f"Failed to process mask: {e}")
                return None
        elif all(hasattr(response, attr) for attr in ['xmin', 'ymin', 'xmax', 'ymax']):
            pixel_x = (response.xmin + response.xmax) // 2
            pixel_y = (response.ymin + response.ymax) // 2
        else:
            rospy.logerr("Response does not contain a valid mask or bounding box.")
            return None
        coordinates = self.get_3d_coordinates(pixel_x, pixel_y)

        if coordinates:
            x, y, z = coordinates
            # Continuously broadcast the TF frame at the computed coordinates
            self.broadcast_tf_frame(x, y, z, response.Class)
            return coordinates
        else:
            rospy.logwarn("No coordinates received.")

    def broadcast_tf_frame(self, x, y, z, name):
        # Create a TF broadcaster
        tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Generate a unique frame name
        frame_name = f"{name}_{uuid.uuid4().hex[:8]}"

        # Define the transform
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = self.parent_frame
        transform.child_frame_id = frame_name

        # Set the translation
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.translation.z = z

        # Set the rotation (no rotation applied, identity quaternion)
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        transform.transform.rotation.w = 1.0

        # Broadcast the transform
        tf_broadcaster.sendTransform(transform)

        rospy.loginfo_once(f"Broadcasting TF frame {frame_name} at x={x:.2f}, y={y:.2f}, z={z:.2f}")

    def get_3d_coordinates(self, pixel_x, pixel_y):
        rospy.wait_for_service('get_3d_coordinates')  # Wait for the service to be available

        try:
            # Create a service proxy
            get_coords_service = rospy.ServiceProxy('get_3d_coordinates', Get3DCoordinates)

            # Create and send the request
            request = Get3DCoordinatesRequest(pixel_x=pixel_x, pixel_y=pixel_y)
            response = get_coords_service(request)

            if response.success:
                # rospy.loginfo(f"3D Coordinates: x={response.x:.2f}, y={response.y:.2f}, z={response.z:.2f}")
                return response.x, response.y, response.z
            else:
                rospy.logwarn("Failed to compute 3D coordinates.")
                return None
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return None
        
    #==================
    # DISPLAY FUNCTIONS
    #------------------
    def display_bounding_boxes(self, image, response):
        """Display bounding boxes from the detection response."""
        for bbox in response.objects.bounding_boxes:
            x_min, y_min, x_max, y_max = int(bbox.xmin), int(bbox.ymin), int(bbox.xmax), int(bbox.ymax)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, f"{bbox.Class} ({round(bbox.probability, 2)})", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the image
        cv2.imshow('Annotated Image', image)
        cv2.waitKey(1)

    def display_segmentation_masks(self, image, response):
        """Display segmentation masks from the segmentation response."""
        for mask in response.objects.masks:
            # Convert the mask to OpenCV format (mono8)
            mask_image = self.bridge.imgmsg_to_cv2(mask.mask, "mono8")

            # Overlay the mask on the image
            colored_mask = cv2.applyColorMap(mask_image, cv2.COLORMAP_JET)
            overlaid_image = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)

            # Find the top-most pixel in the mask for the label
            mask_indices = cv2.findNonZero(mask_image)
            if mask_indices is not None:
                top_left = tuple(mask_indices.min(axis=0)[0])
                text_position = (top_left[0], max(10, top_left[1] - 10))
                cv2.putText(overlaid_image, f"{mask.Class} ({round(mask.probability, 2)})", text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Display the mask
            cv2.imshow('Segmentation Image', overlaid_image)
            cv2.waitKey(1)

    def display_depthmap(self, response):
        """
        Display the depth map from a DepthEstimationResponse.

        Args:
            response (DepthEstimationResponse): The response containing the depth map.

        Returns:
            None
        """
        try:
            # Extract metadata
            width = response.depth_map.width
            height = response.depth_map.height

            # Decode raw byte data if necessary
            if isinstance(response.depth_map.data, bytes):
                depth_data = np.frombuffer(response.depth_map.data, dtype=np.uint8).astype(np.float32)
            else:
                depth_data = np.array(response.depth_map.data, dtype=np.float32)

            # Check if reshaping is possible
            if depth_data.size != width * height:
                rospy.logerr(
                    f"Mismatch in depth data size ({depth_data.size}) and dimensions ({width}x{height})"
                )
                return

            # Reshape to original dimensions
            depth_data = depth_data.reshape((height, width))

            # Normalize depth values for visualization
            max_value = depth_data.max()
            if max_value <= 0:
                rospy.logerr("Invalid max value in depth data. Cannot normalize.")
                return

            formatted_depth = (depth_data * 255 / max_value).astype("uint8")

            # Display the depth map
            cv2.imshow("Depth Estimation", formatted_depth)
            rospy.loginfo("Press any key in the display window to exit.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            rospy.logerr(f"Failed to display depth map: {e}")

    #========================================================================================================
    # OTHER UTILITY FUNCTIONS
    #------------------------

    def query_annotators(self, obs_graph, object):
        """
        Queries the observation graph for annotators able to detect the given object using a SPARQL query.
        Adds a simulated entity of the given type to the ontology, performs reasoning, and then executes the query.
        
        :param obs_graph: The loaded ontology graph (owlready2 ontology object).
        :param object: The target object (assumed to be an IRI or identifier).
        :return: List of results from the SPARQL query, or None if no results are found.
        """
        try:
            rospy.loginfo(f"Adding a simulated entity of type {object} to the ontology...")
            
            # Step 1: Add a simulated entity of type 'object' to the ontology
            # with obs_graph:
            #     obs_graph[object](f"SimulatedEntity_{object}")
            #     # simulated_entity.is_a.append()  # Assign type `object`
            #     rospy.loginfo("Running reasoning...")
            #     sync_reasoner_pellet(infer_property_values=True, debug=0)
            #     rospy.loginfo("Reasoning complete.")

            # Step 2: Construct and run the SPARQL query
            rospy.loginfo(f"Querying the observation graph for annotators capable of detecting {object}...")
            sparql_query_annotators = f"""
            PREFIX sosa: <http://www.w3.org/ns/sosa/>
            PREFIX ssn: <http://www.w3.org/ns/ssn/>
            PREFIX orka: <https://w3id.org/def/orka#>
            PREFIX oboe: <http://ecoinformatics.org/oboe/oboe.1.2/oboe-core.owl#>

            SELECT DISTINCT ?annotator ?annotatorName
            WHERE {{
            # Query for the simulated entity and related annotators
            ?temp_entity a orka:{object} .
            ?annotator orka:canDetect ?temp_entity .
            ?annotator orka:hasServiceName ?annotatorName .
            }}
            """
            results = list(default_world.sparql(sparql_query_annotators, error_on_undefined_entities=False))
            rospy.loginfo(f"SPARQL query returned {len(results)} results.")
            for result in results: print(result[0]) # Print the name of the annotator
            return results if results else None
        
        except Exception as e:
            rospy.logerr(f"Error running query_annotators function: {e}")
            return None

    def transform_coordinates(self, source_frame, target_frame, source_coordinates):
        """
        Transforms coordinates from one TF frame to another.

        Args:
            source_frame (str): Name of the source frame.
            target_frame (str): Name of the target frame.
            source_coordinates (list): [x, y, z] coordinates in the source frame.

        Returns:
            list: Transformed coordinates in the target frame, or None if transformation fails.
        """
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)

        # Create a PointStamped message for the source coordinates
        point = PointStamped()
        point.header.stamp = rospy.Time(0)  # Use the latest available time
        point.header.frame_id = source_frame
        point.point.x = source_coordinates[0]
        point.point.y = source_coordinates[1]
        point.point.z = source_coordinates[2]

        try:
            # Wait for the transform to be available
            rospy.loginfo(f"Waiting for transform from {source_frame} to {target_frame}...")
            if not tf_buffer.can_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(3.0)):
                rospy.logerr(f"Transform from {source_frame} to {target_frame} is not available.")
                return None

            # Perform the transformation
            transformed_point = tf_buffer.transform(point, target_frame)

            # Extract transformed coordinates
            return [transformed_point.point.x, transformed_point.point.y, transformed_point.point.z]

        except tf2_ros.LookupException as e:
            rospy.logerr(f"Transform lookup error: {e}")
        except tf2_ros.ConnectivityException as e:
            rospy.logerr(f"Transform connectivity error: {e}")
        except tf2_ros.ExtrapolationException as e:
            rospy.logerr(f"Transform extrapolation error: {e}")
        except Exception as e:
            rospy.logerr(f"Unexpected error while transforming coordinates: {e}")

        return None

    def query_location(self, obs_graph, object):
        """
        Queries the observation graph for the location of the given object using a SPARQL query.
        :param obs_graph: The loaded ontology graph (owlready2 ontology object).
        :param object: The target object (assumed to be an IRI or identifier).
        :return: List of results from the SPARQL query, or None if no results are found.
        """
        rospy.loginfo(f"Querying the observation graph for the location of {object}...")

        sparql_query_location = f"""
        PREFIX sosa: <http://www.w3.org/ns/sosa/>
        PREFIX ssn: <http://www.w3.org/ns/ssn/>
        PREFIX orka: <https://w3id.org/def/orka#>
        PREFIX oboe: <http://ecoinformatics.org/oboe/oboe.1.2/oboe-core.owl#>

        SELECT ?loc ?ent WHERE {{
        ?ent a orka:{object} .
        ?loc_instance a orka:Location .
        ?loc_instance oboe:characteristicFor ?ent .
        ?loc_instance orka:hasValue ?loc .                               
        }}
        """

        try:
            # Run the SPARQL query on the ontology
            results = list(default_world.sparql(sparql_query_location, error_on_undefined_entities=False))
            rospy.loginfo(f"SPARQL query returned {len(results)} results.")
            return results if results else None
        except Exception as e:
            rospy.logerr(f"Error running SPARQL query: {e}")
            return None


def pickup_object(pickup_coordinates, destination_coordinates):
    """
    Sends a goal to the pickup action server to pick up an object at the given position.
    """
    rospy.loginfo(f"Sending goal to pickup action for position {pickup_coordinates}...")

    # Connect to the action server
    action_client = actionlib.SimpleActionClient('pick_place', PickPlaceAction)
    rospy.loginfo("Waiting for the pick_place action server...")
    action_client.wait_for_server()

    # Wait for the action server to be available
    rospy.loginfo("Waiting for pickup action server...")
    action_client.wait_for_server()

    # Create the goal
    goal = PickPlaceGoal()
    goal.pickup_coordinates = pickup_coordinates
    goal.destination_coordinates = destination_coordinates

    rospy.loginfo(f"Sending pickup coordinates: {pickup_coordinates}")
    rospy.loginfo(f"Sending destination coordinates: {destination_coordinates}")
    
    # Send the goal to the server
    action_client.send_goal(goal)

    # Wait for the result
    rospy.loginfo("Waiting for move action result...")
    action_client.wait_for_result()

    # Retrieve the result
    result = action_client.get_result()
    rospy.loginfo(f"Result: success={result.success}, message={result.message}")


# # Main script
# if __name__ == "__main__":


if __name__ == "__main__":
    try:
        #fruit_salad_items = ['Banana', 'Apple', 'Strawberry', 'Orange', 'Pineapple']
        fruit_salad_items = ['Banana', 'Apple', 'Orange']
        annotator_client = TaskSelector()
        annotator_client.call_service('/annotators/ObjectDetection/yolos_tiny/detect')
        # annotator_client.call_service('/annotators/ImageSegmentation/detr_resnet_50_panoptic/detect')


        for fruit in fruit_salad_items:
            rospy.loginfo(f"Processing {fruit}...")
            fruit_position = None

            obs_graph = annotator_client.get_obs_graph()
            options_left = True

            fruit_position = annotator_client.query_location(obs_graph, fruit)
            capable_annotators = annotator_client.query_annotators(obs_graph, fruit)
            
            while options_left and not fruit_position:
                for annotators in capable_annotators:          
                    annotator_name, service_name = annotators
                    rospy.loginfo(f"Calling service {annotator_name}")
                    annotator_client.call_service(service_name, prompt=fruit)
                    obs_graph = annotator_client.get_obs_graph()
                    fruit_position = annotator_client.query_location(obs_graph, fruit)
                    if fruit_position: break
                    capable_annotators.remove(annotators)
                    if not capable_annotators: options_left = False

            if fruit_position:
                rospy.loginfo(f'{len(fruit_position)} {fruit}s detected!')
                # Define pickup and destination coordinates
                pickup_coordinates = [0.266, 0.075, -0.088]
                pickup_coordinates = [0.266, 0.075, 0.088]
                object_coordinates = [round(float(x), 3) for x in fruit_position[0][0].strip("()").split(",")]
                pickup_coordinates = annotator_client.transform_coordinates('locobot/camera_color_optical_frame', 'locobot/arm_base_link', object_coordinates)
                destination_coordinates = [0, 0.3, 0.2]
                rospy.logwarn(f'Picking up fruit: {fruit_position[0][1]} at position {pickup_coordinates}')
                # pickup_object(pickup_coordinates, destination_coordinates)
                rospy.logwarn(f'---Actual pickup skipped---')

            else:
                rospy.logwarn(f"{fruit} not found!")


        rospy.spin()
    except rospy.ROSInterruptException:
        pass