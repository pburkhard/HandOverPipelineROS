#!/usr/bin/env python

import rospy
from std_msgs.msg import String

def correspondence_estimator():
    # Initialize the node
    rospy.init_node('correspondence_estimator', anonymous=True)

    # Create a publisher to publish correspondence data
    pub = rospy.Publisher('sample_topic', String, queue_size=10)

    # Set the rate at which to publish messages
    rate = rospy.Rate(10)  # 10 Hz

    while not rospy.is_shutdown():
        # Create a message to publish
        msg = String()
        msg.data = "Correspondence data at time: {}".format(rospy.get_time())

        # Publish the message
        pub.publish(msg)

        # Log the message to console
        rospy.loginfo(msg.data)

        # Sleep for the specified rate
        rate.sleep()

if __name__ == '__main__':
    try:
        correspondence_estimator()
    except rospy.ROSInterruptException:
        pass