# Grasp Generator ROS Package

## Setup

1. Rename the file `.env.example` to `.env` and fill in the necessary environment variables.
2. Run `setup.bash`, which will create the necessary conda environment.
3. (Optional) check the `config/default.yaml` file and adapt ROS topic names if necessary.

## Running the node

Use the following command:

```roslaunch grasp_generator grasp_generator.launch```

This will launch the node in the correct conda environment.