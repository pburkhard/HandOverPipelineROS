# Handover Pipeline ROS Package

## Overview

This is the main node of the handover pipeline. It handles the execution of the whole pipeline and calles services or actions of other nodes as needed.


## Setup

1. Run `setup.bash` to create the required conda environment.
2. (Optional) Check the ROS topic names and settings in `config/default.yaml` and change if needed.

## Launching the Node

Launch the node with:

```bash
roslaunch pipeline pipeline.launch
```

This will launch the node in the correct conda environment.

## Using the Node

Please refer to the README at the root of this project for more details on how to configure and run the pipeline.