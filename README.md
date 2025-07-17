# Handover Pipeline

## Overview

This pipeline determins the optimal object orientation in a task-oriented robot-to-human handover scenario. It consists of 6 ROS packages, all running in their individual conda environment to avoid library version conflicts. Here's a short overview on the packages:

- correspondence_estimator: Uses a semantic feature extractor to find correspondece points between objects that are not perfectly identical.
- grasp_generator: Makes calls to the OpenAI API to generate an image depicting a hand grasping a given object in a task-oriented way.
- hand_reconstructor: Uses the vision-transformer-based model "hamer" to estimate the 3D hand pose (and other parameters) from a monocular image.
- motion_executor: Steers a Franka Emika Panda robot arm to a given dynamic target pose.
- pipeline: The main node of the handover pipeline. It handles the execution of the whole pipeline and calles services or actions of other nodes as needed.
- transform_estimator: Applies a robust Perspective-n-Point solving algorithm to recover the transformation from the robot cam coordinate frame to the (fictional) gen cam coordinate frame.

The intended starting point for this pipeline is after a the robot has grasped the object with its gripper, ready to deliver it to a human receiver. The goal is to put the object right into the hand of the receiving human, adapting in realtime to small pose variations. The pipeline first runs an initialization phase, determining how it has to orient the gripper with respect to the hand of the receiver. Then, it enters the main loop where it coninuously tracks the receiving hand and moves the robot gripper to the desired location.

Setup instructions and more detailed explanation of the individual nodes can be found in the corrresponding README's, located at `./handover_pipeline_ws/src/<package_name>/README.md`. Each node has a configuration file containing rostopic names and other settings, located at `./handover_pipeline_ws/src/<package_name>/config/`.

---

## Setup

The steps below explain how to set up the pipeline in a docker container. If you don't want to use it in a container, just ignore step 2.

1. **Initialize all git submodules**

    ```bash
    git submodule update --init --recursive
    ```

2. **Build and enter the Docker container**

    ```bash
    bash ./scripts/start_docker.bash
    ```

    You can attach a VS Code window or exec into the container from the terminal.

3. **(Inside the container) Run the setup script**

    In the workspace root folder, run:

    ```bash
    bash ./handover_pipeline_ws/setup.bash
    ```
    This allows you to use commands like `catkin_make` right out of the box.

4. **(Inside the container) Set up the packages**

    Set up all packages that you want to use by following the steps in their associated README located at `./handover_pipeline_ws/src/<package_name>/README.md`


5. **(Inside the container) Initialize the workspace**

    The conda base environment should have all requirements installed:

    ```bash
    cd ./handover_pipeline_ws
    catkin_make
    ```

---

## Running the Code

### Full pipeline

0. **Source the workspace**

    ```bash
    source /handover_pipeline_ws/devel/setup.bash
    ```

1. **Check the configuration**
    1. Open the pipeline configuration file `./handover_pipeline_ws/src/pipeline/config/default.yaml`
    2. Make sure that all rostopics names are correct and that the required topics are published.
    3. Have a look at the debug section and make sure that none of the nodes or subscribers is bypassed.

2. **Launch the ROS nodes** (each command in a separate terminal)

    ```bash
    roslaunch correspondence_estimator correspondence_estimator.launch
    ```
    ```bash
    roslaunch grasp_generator grasp_generator.launch
    ```
    ```bash
    roslaunch hand_reconstructor hand_reconstructor.launch
    ```
    ```bash
    roslaunch motion_executor motion_executor.launch
    ```
    ```bash
    roslaunch pipeline pipeline.launch
    ```
    ```bash
    roslaunch transform_estimator transform_estimator.launch
    ```
3. **Publish the task topic**: Publish a message containing the tool name and the name of the subsequent task as strings to the topic specified in `cfg.ros.subscribed_topics.task`. The pipeline receives this information and begins to wait for the gripper pose grasping the object.
4. **Publish the gripper pose**: Publish the gripper pose when grasping the object as a `visualisation_msgs/Marker` message to the topic specified in `cfg.ros.subscribed_topics.transform_selected_grasp_to_robot_cam`. This will start the pipeline. Attention: The Marker must be formulated in the coordinate frame of the robot camera.
5. **Trigger the motion execution** Once the pipeline has entered the main loop, trigger the motion execution by sending a message of type `std_msgs/Empty` to the topic specified in `cfg.subscribed_topics.trigger` of the motion executor configuration.

### Bypass nodes and topics

The pipeline is designed such that some nodes can be bypassed, for example due to memory limitations. Data that would have been produced by this node is loaded from a file instead. Checkout the options in the debug section of the pipeline configuration to see what can be bypassed. The folder from which the data will be loaded is specified in `cfg.debug.example_dir`. The file from which the data will be loaded must have a specific name, depending on the node that is bypassed. Please refer to the code to see the exact filenames required in each case, or run the pipeline and investigate the error message when it fails to load a specific file.
Some topics can also be bypassed in the exact same way as nodes.

To run the pipeline with bypassed nodes, proceed exactly in the same way as for the full pipeline, just without starting the bypassed nodes.

---

## Known Issues

### cv_bridge

The `cv_bridge` package (for converting cv2 images to/from ROS images) may cause issues with system libraries:

```
Error: /lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0
```

**Workaround:** Explicitly set the system library path and preload the correct library version:

```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
```

---

### Permission denied

When launching a newly created node, you might see:

```
exec: <path>/<node>.py: cannot execute: Permission denied
```

**Workaround:** Manually add execution permission:

```bash
chmod +x <path>/<node>.py
```