#!/usr/bin/env python3
import rospy
from orvis.srv import Get3DCoordinates, Get3DCoordinatesRequest
from geometry_msgs.msg import TransformStamped
import tf2_ros
import uuid

def get_3d_coordinates(pixel_x, pixel_y):
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

def broadcast_tf_frame_continuously(x, y, z, parent_frame):
    # Create a TF broadcaster
    tf_broadcaster = tf2_ros.TransformBroadcaster()

    # Generate a unique frame name
    frame_name = f"object_{uuid.uuid4().hex[:8]}"

    rate = rospy.Rate(10)  # Broadcast at 10 Hz

    while not rospy.is_shutdown():
        # Define the transform
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = parent_frame
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
        rate.sleep()

if __name__ == "__main__":
    rospy.init_node("get_3d_coordinates_client_with_tf")

    # Example pixel values
    pixel_x, pixel_y = 100, 100
    parent_frame = "locobot/camera_aligned_depth_to_color_frame"

    # Call the service to get 3D coordinates
    coordinates = get_3d_coordinates(pixel_x, pixel_y)

    if coordinates:
        x, y, z = coordinates
        # Continuously broadcast the TF frame at the computed coordinates
        broadcast_tf_frame_continuously(x, y, z, parent_frame)
    else:
        rospy.logwarn("No coordinates received.")
