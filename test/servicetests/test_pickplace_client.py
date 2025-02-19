#!/usr/bin/env python3

import rospy
import actionlib
from orvis.msg import PickPlaceAction, PickPlaceGoal

def main():
    # Initialize the ROS node
    rospy.init_node('pick_and_place_client')

    # Connect to the action server
    client = actionlib.SimpleActionClient('pick_place', PickPlaceAction)
    rospy.loginfo("Waiting for the pick_place action server...")
    client.wait_for_server()

    # Define pickup and destination coordinates
    pickup_coordinates = [0.266, 0.075, -0.088]
    destination_coordinates = [0.271, -0.061, -0.088]

    # Create the goal
    goal = PickPlaceGoal()
    goal.pickup_coordinates = pickup_coordinates
    goal.destination_coordinates = destination_coordinates

    rospy.loginfo(f"Sending pickup coordinates: {pickup_coordinates}")
    rospy.loginfo(f"Sending destination coordinates: {destination_coordinates}")

    # Send the goal to the server
    client.send_goal(goal)

    # Wait for the result
    rospy.loginfo("Waiting for result...")
    client.wait_for_result()

    # Retrieve the result
    result = client.get_result()
    rospy.loginfo(f"Result: success={result.success}, message={result.message}")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

