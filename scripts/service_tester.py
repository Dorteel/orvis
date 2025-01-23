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
import matplotlib.pyplot as plt

def save_results_to_csv(call_times, filename="service_call_times.csv"):
    """
    Save the call times of each service to a CSV file.

    Args:
        call_times (dict): A dictionary where keys are service names and values are lists of call times.
        filename (str): The name of the CSV file to save the results.

    Returns:
        None
    """
    try:
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header: first column is "service_name", followed by individual call numbers
            max_calls = max(len(times) for times in call_times.values())
            header = ["service_name"] + [f"call_{i+1}" for i in range(max_calls)]
            writer.writerow(header)

            # Write each row for the services
            for service_name, times in call_times.items():
                row = [service_name] + times + ["" for _ in range(max_calls - len(times))]  # Pad with empty strings if necessary
                writer.writerow(row)

        rospy.loginfo(f"Call times saved to {filename}")
    except Exception as e:
        rospy.logerr(f"Failed to save call times to CSV: {e}")


def plot_boxplot(call_times, output_file="service_call_times_boxplot.png"):
    """
    Create and save a box plot for the call times of each service.

    Args:
        call_times (dict): A dictionary where keys are service names and values are lists of call times.
        output_file (str): The name of the output image file for the box plot.

    Returns:
        None
    """
    try:
        services = list(call_times.keys())
        times = [call_times[service] for service in services]

        plt.figure(figsize=(10, 6))
        plt.boxplot(times, labels=services, showmeans=True)
        plt.title("Service Call Times Boxplot")
        plt.ylabel("Time (seconds)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_file)

        rospy.loginfo(f"Box plot saved to {output_file}")
    except Exception as e:
        rospy.logerr(f"Failed to create box plot: {e}")


def call_service(service_name, service_type, request, repeat=1):
    """
    Test a ROS service by calling it `repeat` times and measuring performance.

    Args:
        service_name (str): Name of the ROS service to test.
        service_type (type): ROS service type.
        request: The request object to send to the service.
        repeat (int): Number of times to call the service.

    Returns:
        list: List of call times for the service.
    """
    rospy.loginfo(f"Testing service: {service_name} with type: {service_type}")
    call_times = []

    try:
        rospy.wait_for_service(service_name, timeout=5)
        service_proxy = rospy.ServiceProxy(service_name, service_type)

        for i in range(repeat):
            start_time = time.time()
            response = service_proxy(request)
            end_time = time.time()

            elapsed_time = end_time - start_time
            call_times.append(elapsed_time)

            rospy.loginfo(f"  Request {i + 1}/{repeat}: Time elapsed: {elapsed_time:.4f} seconds")

        return call_times
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call to {service_name} failed: {e}")
    except rospy.ROSException as e:
        rospy.logerr(f"Service {service_name} is unavailable: {e}")

    return call_times


def get_annotator_services():
    """
    Retrieve all annotator services available.

    Returns:
        dict: Dictionary of service names and their types.
    """
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
    """
    Callback function for image topic. Tests annotator services.

    Args:
        img_msg (Image): Incoming image message.

    Returns:
        None
    """
    services = get_annotator_services()
    rospy.loginfo(f"Found {len(services)} annotator services.")

    prompt = String()
    prompt.data = "man"
    repeat = 100
    call_times = {}

    for service_name, service_type in services.items():
        rospy.loginfo(f"Testing service: {service_name} with type: {service_type}")
        try:
            if service_type == 'orvis/ObjectDetection':
                request = ObjectDetectionRequest(image=img_msg)
                call_times[service_name] = call_service(service_name, ObjectDetection, request, repeat)
            elif service_type == 'orvis/ImageSegmentation':
                request = ImageSegmentationRequest(image=img_msg)
                call_times[service_name] = call_service(service_name, ImageSegmentation, request, repeat)
            elif service_type == 'orvis/PromptedObjectDetection':
                request = PromptedObjectDetectionRequest(image=img_msg, prompt=prompt)
                call_times[service_name] = call_service(service_name, PromptedObjectDetection, request, repeat)
            elif service_type == 'orvis/VideoClassification':
                video_frames.append(img_msg)
                if len(video_frames) == num_video_frames:
                    request = VideoClassificationRequest(video_frames=video_frames)
                    call_times[service_name] = call_service(service_name, VideoClassification, request, repeat)
            elif service_type == 'orvis/DepthEstimation':
                request = DepthEstimationRequest(image=img_msg)
                call_times[service_name] = call_service(service_name, DepthEstimation, request, repeat)
            elif service_type == 'orvis/ImageToText':
                request = ImageToTextRequest(image=img_msg)
                call_times[service_name] = call_service(service_name, ImageToText, request, repeat)
            else:
                rospy.logwarn(f"Unknown service type: {service_type}")
        except Exception as e:
            rospy.logerr(f"Failed to test service {service_name}: {e}")

    save_results_to_csv(call_times)
    plot_boxplot(call_times)
    rospy.loginfo("Experiment timing finished. Shutting down.")
    rospy.signal_shutdown("Service testing complete.")


if __name__ == "__main__":
    rospy.init_node("service_tester")
    topic = '/locobot/camera/color/image_raw'
    num_video_frames = 16
    rospy.Subscriber(topic, Image, image_callback)
    video_frames = deque(maxlen=num_video_frames)
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
