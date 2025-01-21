#!/home/user/pel_ws/pel_venv/bin/python
import rospy
import rospkg
import actionlib

from orvis.srv import Get3DCoordinates, Get3DCoordinatesRequest
from geometry_msgs.msg import TransformStamped
import tf2_ros
import uuid

from orvis.msg import PickPlaceAction, PickPlaceGoal

import yaml
import cv2

from orvis.srv import ObjectDetection, ObjectDetectionRequest, ObjectDetectionResponse  # Detection service
from orvis.srv import ImageSegmentation, ImageSegmentationRequest, ImageSegmentationResponse  # Segmentation service
from orvis.msg import ImageMasks, ImageMask  # Import the segmentation message types
from orvis.srv import PromptedObjectDetection, PromptedObjectDetectionRequest, PromptedObjectDetectionResponse  # Detection service
from orvis.srv import DepthEstimation, DepthEstimationRequest, DepthEstimationResponse  # Import the necessary service types
from orvis.srv import VideoClassification, VideoClassificationRequest, VideoClassificationResponse  # Detection service
from orvis.srv import ImageToText, ImageToTextRequest, ImageToTextResponse  # Detection service

from std_msgs.msg import String
from sensor_msgs.msg import Image

# from some_msgs.msg import PickupAction, PickupGoal
# from some_msgs.srv import AnnotatorService, AnnotatorServiceRequest
import random
import os
import numpy as np
from datetime import datetime
from collections import deque

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
        self.run_id = 'orvis_orka_' + datetime.now().strftime("%Y%m%d%H%M%S" + '.owl')
        self.save_dir = os.path.join(self.obs_graph_dir, self.run_id)

        # Initialize the ROS service
        self.video_frames = deque(maxlen=self.num_frames)

        

        # Load the ontology
        self.orka = get_ontology(self.orka_path).load()
        self.sosa = self.orka.get_namespace("http://www.w3.org/ns/sosa/")
        self.oboe = self.orka.get_namespace("http://ecoinformatics.org/oboe/oboe.1.2/oboe-core.owl#")
        self.ssn = self.orka.get_namespace("http://www.w3.org/ns/ssn/")

        # Subscribe to the appropriate image topic
        rospy.Subscriber(self.camera_topic, Image, self.image_callback)
        rospy.loginfo("MainAnnotatorClient initialized.")

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

    def call_service(self, service_to_call):
        """
        Call the specified service using the last received image.

        :param service_to_call: A string representing the service to call
                                (e.g., 'ObjectDetection', 'ImageSegmentation', etc.).
        """
        # Wait until at least one image is received
        while not rospy.is_shutdown() and self.last_image is None:
            rospy.loginfo("Waiting for an image...")
            rospy.sleep(0.1)  # Sleep for a short duration to avoid busy-waiting

        
        task_type = service_to_call.split('/')[2]


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
        else:
            raise NameError("Service type not recognized. Check the name of the services.")


        rospy.wait_for_service(service_to_call)
        self.annotator_service = rospy.ServiceProxy(service_to_call, self.service_type)
        rospy.loginfo(f"Service {service_to_call} connected.")

        try:
            # Dispatch the request to the appropriate service processing method
            if task_type == 'ObjectDetection':
                self.process_detection(self.last_image)
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


    def create_obs_graph(self, result):
        """
        Creates an observation graph
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        observation_id = 'obs_' + timestamp  # Format: YYYYMMDDHHMMSS
        obs_instance = self.oboe.Observation(observation_id) # Create an instance of Observation
        
        # Get observation graph if doesn't exist yet
        rospy.loginfo("Fetching observation graph...")

        # Path to the obs_graphs directory (one level up from the script directory)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        obs_graph_dir = os.path.join(os.path.dirname(script_dir), "obs_graphs")

        # Check if the directory exists
        if not os.path.exists(obs_graph_dir):
            rospy.logwarn(f"The directory {obs_graph_dir} does not exist.")
            return None

        # Get all .owl files in the directory
        owl_files = [os.path.join(obs_graph_dir, f) for f in os.listdir(obs_graph_dir) if f.endswith(".owl")]

        # If no .owl files are found, return None
        if owl_files:
            # Find the most recently modified .owl file
            latest_obs_graph_path = max(owl_files, key=os.path.getmtime)
            rospy.loginfo(f"Latest observation graph found: {latest_obs_graph_path}")

            # Load the ontology using owlready2
            self.orka = get_ontology(latest_obs_graph_path).load()
            rospy.loginfo(f"Latest observation graph successfully loaded from {latest_obs_graph_path}.")    
        
        # TODO: Add UsedProcedure Procedure

        if self.service_type == ObjectDetection or self.service_type == PromptedObjectDetection or self.service_type == ImageToText:
            for boundingbox in result.objects.bounding_boxes:
                rospy.loginfo(f'Creating observation for {boundingbox.Class}')
                coordinates = self.create_3d_coordinates(boundingbox)
                rospy.logwarn(f"The coordinates are {coordinates}")
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S") + str(random.randint(1000, 9999))

                # Creating instances
                try:
                    ent_instance = self.orka[boundingbox.Class.capitalize()]('ent_' + timestamp)
                except Exception as e:
                    rospy.loginfo(f"Class \"{boundingbox.Class.capitalize()}\" not found. Defaulting to PhysicalEntity")
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

                # Adding data properties
                result_instance.hasValue.append(str(boundingbox))
                result_instance.hasProbability.append(boundingbox.probability)
                char_instance.hasValue.append("{}, {}".format(int((boundingbox.xmin + boundingbox.xmax)/2), int((boundingbox.ymin + boundingbox.ymax)/2)))

        elif self.service_type == ImageSegmentation:
            for imagemask in result.objects.masks:
                rospy.loginfo(f'Creating observation for {imagemask.Class}')
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S") + str(random.randint(1000, 9999))

                # Creating instances
                try:
                    ent_instance = self.orka[imagemask.Class.capitalize()]('ent_' + timestamp)
                except Exception as e:
                    rospy.loginfo(f"Class \"{imagemask.Class.capitalize()}\" not found. Defaulting to PhysicalEntity")
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

                # Adding data properties
                result_instance.hasValue.append(str(imagemask))
                result_instance.hasProbability.append(imagemask.probability)
                # TODO: Need to calculate middle of the mask
                # char_instance.hasValue.append("{}, {}".format(int((imagemask.xmin + imagemask.xmax)/2), int((imagemask.ymin + imagemask.ymax)/2)))

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

            # Adding data properties
            result_instance.hasValue.append(str(result.activity.data))
            char_instance.hasValue.append(str(result.activity.data))

        self.orka.save(self.save_dir)


#===========================
# IMAGE PROCESSING FUNCTIONS
#---------------------------

    def process_prompteddetection(self, img_msg):
        """Process image detection service requests."""
        try:
            # Convert the ROS Image message to OpenCV format
        # Determine the image encoding and handle accordingly
            if img_msg.encoding == "bgr8":
                # Convert the ROS Image message to OpenCV format for color images
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            elif img_msg.encoding == "16UC1":
                # Convert the ROS Image message to OpenCV format for depth images
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, "16UC1")
            request = PromptedObjectDetectionRequest(image=img_msg, prompt=self.prompt)
            response = self.annotator_service(request)

            # If logging level is DEBUG, display the bounding boxes
            if self.logging_level == 'DEBUG':
                self.display_bounding_boxes(cv_image, response)
            self.create_obs_graph(response)
            return response

        except Exception as e:
            rospy.logerr(f"Error processing detection image: {e}")

    def process_image_to_text(self, img_msg):
        """Process image detection service requests."""
        try:
            # Convert the ROS Image message to OpenCV format
            if img_msg.encoding == "bgr8":
                # Convert the ROS Image message to OpenCV format for color images
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            elif img_msg.encoding == "16UC1":
                # Convert the ROS Image message to OpenCV format for depth images
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, "16UC1")
            request = ImageToTextRequest(image=img_msg)
            response = self.annotator_service(request)

            # If logging level is DEBUG, display the bounding boxes
            if self.logging_level == 'DEBUG':
                self.display_bounding_boxes(cv_image, response)
            self.create_obs_graph(response)
            return response
        except Exception as e:
            rospy.logerr(f"Error processing detection image: {e}")

    def process_detection(self, img_msg):
        """Process image detection service requests."""
        try:
            # Convert the ROS Image message to OpenCV format
            if img_msg.encoding == "bgr8":
                # Convert the ROS Image message to OpenCV format for color images
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            elif img_msg.encoding == "16UC1":
                # Convert the ROS Image message to OpenCV format for depth images
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, "16UC1")
            request = ObjectDetectionRequest(image=img_msg)
            response = self.annotator_service(request)

            # If logging level is DEBUG, display the bounding boxes
            if self.logging_level == 'DEBUG':
                self.display_bounding_boxes(cv_image, response)

            self.create_obs_graph(response)
            return response
        
        except Exception as e:
            rospy.logerr(f"Error processing detection image: {e}")

    def process_segmentation(self, img_msg):
        """Process segmentation service requests."""
        try:
            # Convert the ROS Image message to OpenCV format
            if img_msg.encoding == "bgr8":
                # Convert the ROS Image message to OpenCV format for color images
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            elif img_msg.encoding == "16UC1":
                # Convert the ROS Image message to OpenCV format for depth images
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, "16UC1")
            request = ImageSegmentationRequest(image=img_msg)
            response = self.annotator_service(request)

            # If logging level is DEBUG, display the segmentation masks
            if self.logging_level == 'DEBUG':
                self.display_segmentation_masks(cv_image, response)
            self.create_obs_graph(response)
            return response
        
        except Exception as e:
            rospy.logerr(f"Error processing segmentation image: {e}")

    def process_depthestimation(self, img_msg):
        """Process depth estimation service requests."""
        try:
            # Convert the ROS Image message to OpenCV format
            if img_msg.encoding == "bgr8":
                # Convert the ROS Image message to OpenCV format for color images
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            elif img_msg.encoding == "16UC1":
                # Convert the ROS Image message to OpenCV format for depth images
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, "16UC1")
            request = DepthEstimationRequest(image=img_msg)
            response = self.annotator_service(request)

            # If logging level is DEBUG, display the bounding boxes
            if self.logging_level == 'DEBUG':
                self.display_depthmap(response)
            self.create_obs_graph(response)
            return response
        except Exception as e:
            rospy.logerr(f"Error processing detection image: {e}")

    def process_videoclassification(self, img_msg):
        """Process depth estimation service requests."""
        try:
            # Convert the ROS Image message to OpenCV format
            if img_msg.encoding == "bgr8":
                # Convert the ROS Image message to OpenCV format for color images
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            elif img_msg.encoding == "16UC1":
                # Convert the ROS Image message to OpenCV format for depth images
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, "16UC1")
            self.video_frames.append(cv_image)
            if len(self.video_frames) == self.num_frames:
                ros_video_frames = [self.bridge.cv2_to_imgmsg(frame, "bgr8") for frame in self.video_frames]

                request = VideoClassificationRequest(video_frames=ros_video_frames)
                response = self.annotator_service(request)
                rospy.loginfo(f"Detected activity: {response}")
                self.create_obs_graph(response)
                return response
            
            else:
                rospy.loginfo(f"Collecting frames ({len(self.video_frames)}/{self.num_frames} frames collected)")
        except Exception as e:
            rospy.logerr(f"Error processing detection image: {e}")

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
                pixel_x, pixel_y = np.mean(non_zero_indices, axis=0).astype(int)
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
            self.broadcast_tf_frame(x, y, z)
            return coordinates
        else:
            rospy.logwarn("No coordinates received.")

    def broadcast_tf_frame(self, x, y, z):
        # Create a TF broadcaster
        tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Generate a unique frame name
        frame_name = f"object_{uuid.uuid4().hex[:8]}"

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
                rospy.loginfo(f"3D Coordinates: x={response.x:.2f}, y={response.y:.2f}, z={response.z:.2f}")
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
def query_annotators(obs_graph, object):
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
        with obs_graph:
            simulated_entity = obs_graph[object](f"SimulatedEntity_{object}")
            # simulated_entity.is_a.append()  # Assign type `object`
            rospy.loginfo("Running reasoning...")
            sync_reasoner_pellet(infer_property_values=True, debug=0)
            rospy.loginfo("Reasoning complete.")

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

def get_obs_graph():
    """
    Fetches and loads the most recently modified observation graph (.owl file) 
    from the knowledge base directory using owlready2.
    Returns the loaded ontology or None if no .owl file exists.
    """
    rospy.loginfo("Fetching observation graph...")

    # Path to the obs_graphs directory (one level up from the script directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    obs_graph_dir = os.path.join(os.path.dirname(script_dir), "obs_graphs")

    # Check if the directory exists
    if not os.path.exists(obs_graph_dir):
        rospy.logwarn(f"The directory {obs_graph_dir} does not exist.")
        return None

    # Get all .owl files in the directory
    owl_files = [os.path.join(obs_graph_dir, f) for f in os.listdir(obs_graph_dir) if f.endswith(".owl")]

    # If no .owl files are found, return None
    if not owl_files:
        rospy.logwarn(f"No .owl files found in {obs_graph_dir}.")
        return None

    # Find the most recently modified .owl file
    latest_obs_graph_path = max(owl_files, key=os.path.getmtime)
    rospy.loginfo(f"Latest observation graph found: {latest_obs_graph_path}")

    # Load the ontology using owlready2
    try:
        ontology = get_ontology(latest_obs_graph_path).load()
        rospy.loginfo(f"Ontology successfully loaded from {latest_obs_graph_path}.")
        return ontology
    except Exception as e:
        rospy.logerr(f"Failed to load ontology: {e}")
        return None


def query_location(obs_graph, object):
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

def call_annotator(annotator_service, object):
    """
    Calls the annotator service to detect the given object.
    """
    rospy.loginfo(f"Calling annotator service '{annotator_service}' to detect {object}...")
    try:
        # Call the service
        annotator_client.call_service(annotator_service)

    except rospy.ServiceException as e:
        rospy.logerr(f"Failed to call annotator service '{annotator_service}': {e}")


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

            obs_graph = get_obs_graph()

            options_left = True

            fruit_position = query_location(obs_graph, fruit)
            capable_annotators = query_annotators(obs_graph, fruit)
            
            while options_left and not fruit_position:
                for annotators in capable_annotators:          
                    annotator_name, service_name = annotators
                    rospy.loginfo(f"Calling service {annotator_name}")
                    annotator_client.call_service(service_name)
                    obs_graph = get_obs_graph()
                    fruit_position = query_location(obs_graph, fruit)

                    if fruit_position:
                        break

                    capable_annotators.remove(annotators)
                    if not capable_annotators:
                        options_left = False

            if fruit_position:
                # Define pickup and destination coordinates
                pickup_coordinates = [0.266, 0.075, -0.088]
                destination_coordinates = [0.271, -0.061, -0.088]
                rospy.loginfo(f'Picking up fruit{fruit}')
                # pickup_object(pickup_coordinates, destination_coordinates)

            else:
                rospy.logwarn(f"{fruit} not found!")


        rospy.spin()
    except rospy.ROSInterruptException:
        pass