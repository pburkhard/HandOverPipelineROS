# Hand Reconstructor ROS Package

## Overview

The grasp generator package is part of the handover pipeline. It uses the vision-transformer-based model "hamer" to estimate the 3D hand pose (and other parameters) from a monocular image. More information on hamer can be found at https://geopavlakos.github.io/hamer/. To get further information on how this node is used in the context of the whole pipeline, please consult the README at the root of this repository.


## Setup

1. Prepare the model data:

    1. Download the hamer data:

        ```bash
        wget https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz
        ```

    2. Unzip the downloaded folder and extract the content to `.data/hamer/` at the root of this repository. It should look something like this:
    
        ```
        HandOverPipelineROS/
            └── data/
                └── ...
                └── hamer/_DATA/
                    ├── data/
                    │   └── mano/
                    │   └── mano_mean_params.npz
                    ├── hamer_ckpts/
                    │   ├── checkpoints/
                    │   │   └── hamer.ckpt
                    │   ├── dataset_config.yaml
                    │   └── model_config.yaml
                    └── vitpose_ckpts/vitpose+_huge/
                        └── wholebody.pth
        ```
    
        > **Note:** The path specified above corresponds to `/Handover_Pipeline/data/hamer/` inside the docker container.

    3. Download the MANO model from the official website: https://mano.is.tue.mpg.de/download.php. If you don't have an account already, you have to create one first. Only the right hand model is needed.

    4. Put the `MANO_RIGHT.pkl` file into the folder `.data/hamer/_DATA/data/mano/`.

2. Run `setup.bash`, which will create the necessary conda environment.

3. (Optional) Check the ROS topic names in `config/default.yaml` and change if needed.

## Launching the Node
Launch the node with:

```bash
roslaunch hand_reconstructor hand_reconstructor.launch
```

This will launch the node in the correct conda environment.


## Using the Node

The node provides multiple service proxies accepting requests of type `EstimateCameraRequest`, `ReconstructHandRequeset`, `ReconstructHandPoseRequest` and `RenderHandRequest`. For more details, see the corresponding `.srv` files located in the `srv` folder of this package.


## Known Issues

### Numpy data type
The installation of hamer as described in their README leads to a faulty environment. When trying to run the code, the following error shows up:

```bash
ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
```

**Workaround:** As described in the official issues (https://github.com/geopavlakos/hamer/issues/106), the issue is that numpy 2.* is not compatible with the code. So just install the latest numpy 1.* version:

```bash
conda activate hand_reconstructor
pip install --upgrade numpy==1.26.4
```

## Pyrenderer Error
The graphics interface EGL gives the following error when running the code in a docker container:

```bash
Error: Invalid device ID (0)
```

**Workaround:** First, try to explicitely set the device id environment variable before running the node:

```bash
EGL_DEVICE_ID=0
CUDA_VISIBLE_DEVICES=0
```

If this doesn't work, install a software-based interface (slower):

```bash
apt-get update && apt-get install -y libosmesa6-dev
```