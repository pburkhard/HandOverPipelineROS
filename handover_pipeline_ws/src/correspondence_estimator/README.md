# Correspondence Estimator ROS Package

## Overview

The correspondence estimator package is part of the handover pipeline. It uses a semantic feature extractor to find correspondence points between objects that are not perfectly identical. More information on the feature extractor can be found here: https://dino-vit-features.github.io/. To get further information on how this node is used in the context of the whole pipeline, please consult the README at the root of this repository.

## Setup

1. Run `setup.bash` to create the required conda environment.
2. (Optional) Check the ROS topic names in `config/default.yaml` and change if needed.

## Launching the Node

Launch the node with:

```bash
roslaunch correspondence_estimator correspondence_estimator.launch
```

This will launch the node in the correct conda environment.

## Using the Node

The node provides an action server that accepts goals of type `EstimateCorrespondenceGoal`. For more details, see the corresponding `.action` file located in the `action` folder of this package.