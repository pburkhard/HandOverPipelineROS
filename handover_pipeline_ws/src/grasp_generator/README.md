# Grasp Generator ROS Package

## Overview

The grasp generator package is part of the handover pipeline. It makes calls to the OpenAI API to generate an image depicting a hand grasping a given object in a task-oriented manner. To get further information on how this node is used in the context of the whole pipeline, please consult the README at the root of this repository.

## Setup

1. Rename the file `.env.example` to `.env` and fill in the necessary environment variables.
2. Run `setup.bash` to create the required conda environment.
3. (Optional) Check the ROS topic names in `config/default.yaml` and change if needed.

## Launching the Node
Launch the node with:

```bash
roslaunch grasp_generator grasp_generator.launch
```

This will launch the node in the correct conda environment.

## Using the Node

The node provides an action server that accepts goals of type `GenerateGraspGoal`. For more details, see the corresponding `.action` file located in the `action` folder of this package.