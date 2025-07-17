# Transform Estimator ROS Package

## Overview

The transform estimator package is part of the handover pipeline. It applies a robust Perspective-n-Point solving algorithm to recover the transformation from the robot cam coordinate frame to the (fictional) gen cam coordinate frame. To get further information on how this node is used in the context of the whole pipeline, please consult the README at the root of this repository.

## Setup

1. Run `setup.bash` to create the required conda environment.
2. (Optional) Check the ROS topic names in `config/default.yaml` and change if needed.

## Launching the Node
Launch the node with:

```bash
roslaunch transform_estimator transform_estimator.launch
```

This will launch the node in the correct conda environment.

## Using the Node

The node provides two action servers that accepts goals of type `EstimateTransformGoal` and `EstimateTransformHeuristicGoal`. For more details, see the corresponding `.action` files located in the `action` folder of this package.