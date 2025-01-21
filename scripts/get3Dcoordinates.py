#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import CameraInfo, Image
from orvis.srv import Get3DCoordinates, Get3DCoordinatesResponse
from cv_bridge import CvBridge
import numpy as np

class Get3DCoordinatesService:
    def __init__(self):
        rospy.init_node('get_3d_coordinates_service')

        # Camera parameters
        self.bridge = CvBridge()
        self.depth_image = None
        self.camera_intrinsics = None
        self.depth_scale = 0.001  # Convert depth from mm to meters (if necessary)

        # Subscribers
        rospy.Subscriber('/locobot/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        rospy.Subscriber('/locobot/camera/aligned_depth_to_color/camera_info', CameraInfo, self.camera_info_callback)

        # Service
        self.service = rospy.Service('get_3d_coordinates', Get3DCoordinates, self.handle_request)
        rospy.loginfo("3D Coordinate Service is ready.")

    def depth_callback(self, msg):
        # Convert depth image to numpy array
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def camera_info_callback(self, msg):
        # Store camera intrinsics
        self.camera_intrinsics = {
            'fx': msg.K[0],  # Focal length x
            'fy': msg.K[4],  # Focal length y
            'cx': msg.K[2],  # Optical center x
            'cy': msg.K[5]   # Optical center y
        }
        # rospy.loginfo(f"Camera Intrinsics received: {self.camera_intrinsics}")

    def handle_request(self, req):
        if self.depth_image is None or self.camera_intrinsics is None:
            rospy.logwarn("Depth image or camera intrinsics are not yet available.")
            return Get3DCoordinatesResponse(success=False, x=0, y=0, z=0)

        u, v = req.pixel_x, req.pixel_y

        # Validate pixel coordinates
        if not (0 <= u < self.depth_image.shape[1] and 0 <= v < self.depth_image.shape[0]):
            rospy.logwarn(f"Pixel coordinates ({u}, {v}) are out of bounds for the depth image.")
            return Get3DCoordinatesResponse(success=False, x=0, y=0, z=0)

        # Retrieve depth value
        depth = self.depth_image[v, u] * self.depth_scale  # Convert to meters
        if depth == 0 or np.isnan(depth):
            rospy.logwarn(f"Invalid depth at pixel ({u}, {v}).")
            return Get3DCoordinatesResponse(success=False, x=0, y=0, z=0)

        # Retrieve camera intrinsics
        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy']
        cx = self.camera_intrinsics['cx']
        cy = self.camera_intrinsics['cy']

        # Compute 3D coordinates
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        rospy.loginfo(f"Computed 3D coordinates: x={x:.2f}, y={y:.2f}, z={z:.2f} for pixel ({u}, {v})")
        return Get3DCoordinatesResponse(success=True, x=x, y=y, z=z)

if __name__ == '__main__':
    try:
        service = Get3DCoordinatesService()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
