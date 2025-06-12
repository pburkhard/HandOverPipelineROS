# Handover Pipeline

## Setup

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

### Quickstart

0. **Source the workspace**

    ```bash
    source /handover_pipeline_ws/devel/setup.bash
    ```

1. **Run the desired ROS node**

    ```bash
    roslaunch <ROS package> <ROS node>.launch
    ```

    To run the main pipeline node:

    ```bash
    roslaunch pipeline pipeline.launch
    ```

2. **Consult individual package documentation**

    - See `./handover_pipeline_ws/src/<package_name>/README.md`
    - Check configuration in `./handover_pipeline_ws/src/<package_name>/config/`

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