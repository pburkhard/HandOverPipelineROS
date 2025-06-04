#!/usr/bin/env python
import rospy
from std_msgs.msg import String

def callback(data):
    rospy.loginfo("Received data: %s", data.data)

def grasp_generator():
    # Initialize the node
    rospy.init_node('grasp_generator', anonymous=True)

    # Create a subscriber to listen for messages on 'sample_topic'
    rospy.Subscriber('sample_topic', String, callback)

    rospy.spin()

if __name__ == '__main__':
    grasp_generator()