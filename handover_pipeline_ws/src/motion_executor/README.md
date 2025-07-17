# Motion Executor ROS Package

## Overview

The motion executor package is part of the handover pipeline. It steers a Franka Emika Panda robot arm to a given dynamic target pose. To get further information on how this node is used in the context of the whole pipeline, please consult the README at the root of this repository.

## Setup

1. Run `setup.bash` to create the required conda environment.
2. (Optional) Check the ROS topic names in `config/default.yaml` and change if needed.

## Launching the Node
Launch the node with:

```bash
roslaunch motion_executor motion_executor.launch
```

This will launch the node in the correct conda environment.

## Using the Node

- Publish the target gripper pose in a message of type `geometry_msgs/PoseStamped` to the topic specified in `cfg.ros.subscribed_topics.target_pose`. The topic must be published in a continuous loop.
- Publish a message of type `std_msgs/Empty` to the topic specified in `cfg.subscribed_topics.trigger`. This will trigger the motion execution. The topic must only be published once.


Note: The robot won't move until it receives the trigger message!