#!/usr/bin/env python

import rospy
import time
from orvis.srv import ObjectDetection, ObjectDetectionRequest, ObjectDetectionResponse  # Detection service
from orvis.srv import ImageSegmentation, ImageSegmentationRequest, ImageSegmentationResponse  # Segmentation service
from orvis.srv import PromptedObjectDetection, PromptedObjectDetectionRequest, PromptedObjectDetectionResponse  # Detection service
from orvis.srv import DepthEstimation, DepthEstimationRequest, DepthEstimationResponse  # Import the necessary service types
from orvis.srv import VideoClassification, VideoClassificationRequest, VideoClassificationResponse  # Detection service
from orvis.srv import ImageToText, ImageToTextRequest, ImageToTextResponse  # Detection service
from collections import deque

import rosservice
from std_msgs.msg import String
from sensor_msgs.msg import Image
import importlib
import csv

def save_results_to_csv(results, filename="service_test_results.csv"):
    """
    Save the results of service testing to a CSV file.

    Args:
        results (list): A list of dictionaries with service testing details.
                        Each dictionary should have keys like 'service_name',
                        'service_type', 'total_time', 'avg_time', 'calls_per_second'.
        filename (str): The name of the CSV file to save the results.

    Returns:
        None
    """
    if not results:
        rospy.logwarn("No results to save.")
        return

    # Extract the header from the first result dictionary
    headers = results[0].keys()

    try:
        with open(filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results)

        rospy.loginfo(f"Results saved to {filename}")
    except Exception as e:
        rospy.logerr(f"Failed to save results to CSV: {e}")



def call_service(service_name, service_type, request, repeat=1):
    """
    Test a ROS service by calling it `repeat` times and measuring performance.

    Args:
        service_name (str): Name of the ROS service to test.
        service_type (type): ROS service type.
        request: The request object to send to the service.
        repeat (int): Number of times to call the service.

    Returns:
        None
    """
    rospy.loginfo(f"Testing service: {service_name} with type: {service_type}")
    result = {
            "service_name": service_name,
            "service_type": service_type,
        }
    try:
        rospy.wait_for_service(service_name, timeout=5)
        service_proxy = rospy.ServiceProxy(service_name, service_type)
        total_time = 0

        for i in range(repeat):
            start_time = time.time()
            response = service_proxy(request)
            end_time = time.time()

            elapsed_time = end_time - start_time
            total_time += elapsed_time

            rospy.loginfo(f"  Request {i + 1}/{repeat}: Time elapsed: {elapsed_time:.4f} seconds")

        avg_time = total_time / repeat
        calls_per_second = 1 / avg_time if avg_time > 0 else 0

        rospy.loginfo(f"Summary for {repeat} calls to {service_name}:")
        rospy.loginfo(f"  Total time: {total_time:.4f} seconds")
        rospy.loginfo(f"  Average time per call: {avg_time:.4f} seconds")
        rospy.loginfo(f"  Calls per second: {calls_per_second}")
        
        result.update({
            "total_time": total_time,
            "avg_time": avg_time,
            "calls_per_second": calls_per_second
        })
        return result
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call to {service_name} failed: {e}")
    except rospy.ROSException as e:
        rospy.logerr(f"Service {service_name} is unavailable: {e}")


def get_annotator_services():
    all_services = rosservice.get_service_list()
    service_dict = {}
    for service in all_services:
        if 'annotator' in service:
            try:
                service_type = rosservice.get_service_type(service)
                service_dict[service] = service_type
            except Exception as e:
                rospy.logwarn(f"Could not fetch type for service {service}: {e}")
    return service_dict


def image_callback(img_msg):
    services = get_annotator_services()
    rospy.loginfo(f"Found {len(services)} annotator services.")

    prompt = String()
    prompt.data = "man"
    repeat = 10
    results = []

    for service_name, service_type in services.items():
        result = {
                "service_name": service_name,
                "service_type": service_type,
            }
        rospy.loginfo(f"Testing service: {service_name} with type: {service_type}")
        try:
            if service_type == 'orvis/ObjectDetection':
                request = ObjectDetectionRequest(image=img_msg)
                result = call_service(service_name, ObjectDetection, request, repeat)
            elif service_type == 'orvis/ImageSegmentation':
                request = ImageSegmentationRequest(image=img_msg)
                result = call_service(service_name, ImageSegmentation, request, repeat)
            elif service_type == 'orvis/PromptedObjectDetection':
                request = PromptedObjectDetectionRequest(image=img_msg, prompt=prompt)
                result = call_service(service_name, PromptedObjectDetection, request, repeat)
            elif service_type == 'orvis/VideoClassification':
                video_frames.append(img_msg)
                if len(video_frames) == num_video_frames:
                    request = VideoClassificationRequest(video_frames=video_frames)
                    result = call_service(service_name, VideoClassification, request, repeat)
            elif service_type == 'orvis/DepthEstimation':
                request = DepthEstimationRequest(image=img_msg)
                result = call_service(service_name, DepthEstimation, request, repeat)
            elif service_type == 'orvis/ImageToText':
                request = ImageToTextRequest(image=img_msg)
                result = call_service(service_name, ImageToText, request, repeat)
            else:
                rospy.logwarn(f"Unknown service type: {service_type}")
        except Exception as e:
            rospy.logerr(f"Failed to test service {service_name}: {e}")
        results.append(result)

    if service_type != 'orvis/VideoClassification':
        save_results_to_csv(results)
        rospy.loginfo("Experiment timing finished. Shutting down.")
        rospy.signal_shutdown("Service testing complete.")


if __name__ == "__main__":
    rospy.init_node("service_tester")
    topic = '/webcam/image_raw'
    num_video_frames = 16
    rospy.Subscriber(topic, Image, image_callback)
    video_frames = deque(maxlen=num_video_frames)
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


