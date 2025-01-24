#!/usr/bin/env python

import os
import subprocess
import rospy

def record_rosbag():
    # Define the topic, output directory, and bag file name
    topic = "/locobot/camera/color/image_raw"
    output_directory = "/home/user/Videos/orvis"
    bag_name = "orvis_experiment.bag"
    output_path = os.path.join(output_directory, bag_name)

    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        rospy.loginfo(f"Created directory: {output_directory}")

    rospy.loginfo("Starting rosbag recording...")
    try:
        # Use subprocess to run the rosbag record command
        subprocess.run(
            ["rosbag", "record", "-O", output_path, topic],
            check=True
        )
    except subprocess.CalledProcessError as e:
        rospy.logerr(f"Failed to record rosbag: {e}")
    except KeyboardInterrupt:
        rospy.loginfo("Recording interrupted by user.")
    finally:
        rospy.loginfo(f"Rosbag saved to {output_path}")

if __name__ == "__main__":
    rospy.init_node("rosbag_recorder", anonymous=True)
    try:
        record_rosbag()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node interrupted.")
