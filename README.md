# FishBot MultiRobot Simulator

![Gazebo screenshot](doc/imgs/simulation_launched.png)

Multi-robot simulation stack for testing lidar-inertial SLAM, map fusion, and navigation on a swarm of robots. Everything runs in Docker with ROS 2 Humble.

## What’s inside

- Gazebo-based simulator (`src/gazebo_sim`) with parametrised robot blueprints and spawn helpers.
- Swarm-LIO2 ROS 2 port (`src/Swarm-LIO2-ROS2-Docker`) for multi-agent lidar-inertial odometry.
- Map fusion (`src/map_fusion`) that merges per-robot point clouds into `/global_downsampled_map` and republishes global poses.
- ESDF map (`src/esdf_map`) that builds a 3D ESDF from either fused or per-robot clouds and exposes grid topics plus a distance/gradient query service.
- Foxglove wrapper (`src/foxglove_app`) for multi-agent visualization layouts.
- Navigation controller (`src/rover3d_navigation`). Not integrated yet.
- Exploration module: ***TODO***

## How the pieces fit together

![Frame graph](doc/imgs/frames.png)

- **Gazebo** spawns `botN` robots (default `bot1..bot4`) with URDFs built from YAML configs. Each robot publishes odom/IMU/lidar topics into ROS 2 and TF frames for every robot.
- **Swarm-LIO2** reads lidar/IMU per robot for ego and other agents state estimation (see [`src/Swarm-LIO2-ROS2-Docker/README.md`](src/Swarm-LIO2-ROS2-Docker/README.md)). After SLAM initializes it publishes extrinsic transforms between agents.
- **Map fusion** listens to `map_origin -> botN/world` TF, accumulates, and publishes a merged global map plus each agent pose relative to `map_origin`.
- **ESDF map** consumes either the fused map (`/global_downsampled_map`) or per-robot `/botX/cloud_registered` and publishes `/esdf/grid`, `/esdf/grid_roi`, `/esdf/costmap_2d`, plus the `/esdf/query` distance/gradient service for planners.
- **Foxglove** provides ready layouts for multi-agent visualization.
- **Navigation** *TODO*.
- **Exploration** *TODO*.

### Interfaces for autonomous exploration

- Control input: `/bot*/cmd_vel` (geometry_msgs/Twist) per robot namespace.
- Shared map: `/global_downsampled_map` (sensor_msgs/PointCloud2) fused from all agents.
- Global odometry: `/bot*/global_odom` (nav_msgs/Odometry) relative to `map_origin` for localization.
- ESDF map: `/esdf/grid` (full) + `/esdf/grid_roi` (updated region) in `map_origin`, 2D slice `/esdf/costmap_2d`, and query service `/esdf/query`.

## Repository structure

- `docker/` – Dockerfiles per module, shared entrypoint, and `ros.env` for ROS domain/implementation.
- `docker-compose.yml` – top-level entrypoint to launch containers.
- `src/` – source code for each package:
  - `gazebo_sim/` – Gazebo worlds, robot blueprints, and spawn launch files.
  - `Swarm-LIO2-ROS2-Docker/` – upstream SLAM port (includes its own compose file).
  - `map_fusion/` – map fusion nodes.
  - `esdf_map/` – ESDF builder node, launch, and config.
  - `rover3d_navigation/` – navigation package.
  - `foxglove_app/` – Foxglove bridge wrapper, layouts, and optional extension.

## Prerequisites

- **Docker** + **docker compose v2**.
- NVIDIA GPU with drivers + `nvidia-container-toolkit` (Gazebo, Swarm-LIO2, and map fusion use `runtime: nvidia`/`gpus: all`).

## Clone the repo (with submodules)

```bash
git clone --recursive https://github.com/Vor-Art/fishbot_multirobot_sim.git
cd fishbot_multirobot_sim

# If you forget --recursive:
git submodule update --init --recursive
```

Note: Right now you need to manually switch SLAM submodule branch to actual for this project:

```
cd src/Swarm-LIO2-ROS2-Docker
git checkout ros2_testing_2d_extrinsics
```

## Quick start (everything via Docker)

![Simulation demo](doc/imgs/simulator_demo.gif)

1. Build images (base + modules):

   ```bash
   docker compose build
   ```

   builds all Docker images for this project.

2. Allow GUI access for Gazebo/RViz (if needed):

   ```bash
   xhost +local:root
   ```

3. Launch the full stack (Gazebo + SLAM + fusion + ESDF + Foxglove):

   ```bash
   docker compose up gazebo swarm_lio2 map_fusion esdf_map foxglove
   ```

   - Gazebo: spawns `FISHBOT_AGENT_COUNT` robots on a circle (defaults to 4, names `bot1..botN`).
   - Swarm-LIO2: runs its simulation launch file with `bot_count:=FISHBOT_AGENT_COUNT`.
   - Map fusion: outputs `/global_downsampled_map` and `/bot*/global_pose`.
   - ESDF map: consumes fused or per-robot clouds (configurable) and publishes `/esdf/grid`, `/esdf/grid_roi`, `/esdf/costmap_2d`, and `/esdf/query`.
   - Foxglove: launches server on `ws://localhost:8765`.

4. Inspect topics if you want (inside any container):

   ```bash
   docker exec -it fishbot_gazebo bash
   ros2 topic list
   ```

## Configuring number of agents

- Edit the top-level `.env` (auto-read by Docker Compose) to set the swarm size once:

  ```bash
  # .env
  FISHBOT_AGENT_COUNT=6   # spawns bot1..bot6 in Gazebo, SLAM, and Foxglove
  ```

- You can override per shell if needed:

  ```bash
  FISHBOT_AGENT_COUNT=2 docker compose up swarm_lio2
  ```

**Note:** The compiled maximum swarm size is set in `src/Swarm-LIO2-ROS2-Docker/src/swarm_lio/include/common_lib.h` as `MAX_DRONE_ID` (currently 20). Increasing it enlarges the EKF state (`DIM_STATE`) and can add significant compute/memory cost, keep it only as high as you need.

## Slowing down simulation time

- **Why:** when many agents run individual SLAM pipelines, processing a frame can exceed the nominal Gazebo frame period. Slowing sim time keeps `/clock` behind wall time so SLAM nodes have enough compute to stay in sync.

- Set `FISHBOT_SIM_TIME_SCALE` in `.env` to scale Gazebo’s real-time factor/update rate (e.g., `0.5` runs sim time at half speed to give per-robot SLAM more CPU headroom):

  ```bash
  # .env
  FISHBOT_SIM_TIME_SCALE=0.5
  ```

- Override per shell if you want to experiment:

  ```bash
  FISHBOT_SIM_TIME_SCALE=0.25 docker compose up gazebo
  ```

## How to initialize SLAM

Goal: move one agent until all others initialize.
Note: wait about 10 seconds to IMU initialization for SLAM, before moving.

- **Step 1:** enter the Gazebo container shell.

    ```bash
    docker exec -it fishbot_gazebo bash
    ```

- **Step 2:** start keyboard teleop:

    ```bash
    ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args --remap cmd_vel:=/bot1/cmd_vel
    ```

- **Step 3:** move the agent around the others until all are detected (green/yellow cluster). Use RViz for easier visualization.

## TODOs

- Fix problems with many agents 
