#!/usr/bin/env python

import rospy
import time
from orvis.srv import ObjectDetection, ObjectDetectionRequest, ObjectDetectionResponse  # Detection service
from orvis.srv import ImageSegmentation, ImageSegmentationRequest, ImageSegmentationResponse  # Segmentation service
from orvis.srv import PromptedObjectDetection, PromptedObjectDetectionRequest, PromptedObjectDetectionResponse  # Detection service
import rosservice
import importlib

def call_service(service_name, service_type, request, repeat=1):
    """
    Test a ROS service by calling it `repeat` times and measuring performance.

    Args:
        service_name (str): Name of the ROS service to test.
        service_type (type): ROS service type (e.g., ImageDetection, Empty).
        request: The request object to send to the service.
        repeat (int): Number of times to call the service.

    Returns:
        None
    """
    rospy.loginfo(f"Testing service: {service_name}")
    rospy.wait_for_service(service_name)

    try:
        service_proxy = rospy.ServiceProxy(service_name, service_type)
        total_time = 0
        for i in range(repeat):
            start_time = time.time()
            response = service_proxy(request)
            end_time = time.time()

            elapsed_time = end_time - start_time
            total_time += elapsed_time

            rospy.loginfo(f"Request {i + 1}/{repeat}:")
            rospy.loginfo(f"  Request: {request}")
            rospy.loginfo(f"  Response: {response}")
            rospy.loginfo(f"  Time elapsed: {elapsed_time:.4f} seconds")

        avg_time = total_time / repeat
        rospy.loginfo(f"Summary for {repeat} calls:")
        rospy.loginfo(f"  Total time: {total_time:.4f} seconds")
        rospy.loginfo(f"  Average time per call: {avg_time:.4f} seconds")
        rospy.loginfo(f"  Calls per second: {1 / avg_time if avg_time > 0 else 0:.2f}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")


def get_services():
    all_services = rosservice.get_service_list()
    service_dict = {}
    for service in all_services:
        if 'annotator' in service:
            service_type = service.split('/')[2]
            service_name = service.split('/')[3]
            try:
                service_dict[service_type].append(service_name)
            except KeyError:
                service_dict[service_type] = [service_name]
    for k, v in service_dict.items():
        print(k, v)
    
    return service_dict

if __name__ == "__main__":
    rospy.init_node("service_tester")

    # Configuration: Update the service name and type
    # service_name = "/owl_v2/detect"  # Replace with your service name
    # service_type = Empty  # Replace with your service type
    # request = Empty._request_class()  # Replace with your service request object
    repeat = 10  # Number of times to call the service

    # Run the service tester
    # call_service(service_name, service_type, request, repeat)
