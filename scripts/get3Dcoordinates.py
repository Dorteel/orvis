#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo, Image
from orvis.srv import Get3DCoordinates, Get3DCoordinatesResponse
from cv_bridge import CvBridge

class Get3DCoordinatesService:
    def __init__(self):
        rospy.init_node('get_3d_coordinates_service')

        # Camera parameters
        self.bridge = CvBridge()
        self.depth_image = None
        self.camera_intrinsics = None

        rospy.Subscriber('/locobot/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        rospy.Subscriber('/locobot/camera/aligned_depth_to_color/camera_info', CameraInfo, self.camera_info_callback)

        self.service = rospy.Service('get_3d_coordinates', Get3DCoordinates, self.handle_request)
        rospy.loginfo("3D Coordinate Service is ready.")

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def camera_info_callback(self, msg):
        self.camera_intrinsics = {
            'fx': msg.K[0],
            'fy': msg.K[4],
            'cx': msg.K[2],
            'cy': msg.K[5]
        }

    def handle_request(self, req):
        if self.depth_image is None or self.camera_intrinsics is None:
            rospy.logwarn("Depth image or camera intrinsics are not yet available.")
            return Get3DCoordinatesResponse(success=False, x=0, y=0, z=0)

        u, v = req.pixel_x, req.pixel_y
        depth = self.depth_image[v, u]

        if depth == 0 or depth is None:
            rospy.logwarn(f"Invalid depth at pixel ({u}, {v}).")
            return Get3DCoordinatesResponse(success=False, x=0, y=0, z=0)

        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy']
        cx = self.camera_intrinsics['cx']
        cy = self.camera_intrinsics['cy']

        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        return Get3DCoordinatesResponse(success=True, x=x, y=y, z=z)

if __name__ == '__main__':
    try:
        service = Get3DCoordinatesService()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
