# Fishbot Navigation Module

This package hosts a lightweight per-robot navigation controller (`rover3d_navigation`).

## 目录结构

- `src/` — 节点入口（navigator.py, planning_apf_node.py, gmm_goal_publisher_node.py）
- `include/rover3d_navigation/` — 库代码（ROVER_3D, control_law_3D, Planning_3D, esdf_adapter, init_Graph_CVaR_3D, init_scene_3D, CVaR_SDF_constraint_3D）
- `config/` — 配置文件
- `launch/` — Launch 文件
It assumes every robot provides the topics `/ROBOT/odom` and `/ROBOT/cmd_vel` plus a
`/ROBOT/move_base/goal` topic where RViz publishes 2D Nav Goals.

## Features

- Goal-oriented proportional controller with heading alignment and slowdown near the waypoint.
- Works with arbitrary robot namespaces (e.g. `bot1`, `bot2`, ...).
- Publishes `/ROBOT/nav_path` (for RViz) and `/ROBOT/nav_status` (text state machine output).
- `/ROBOT/cancel_navigation` service instantly stops the robot.

## Topics & services

- Subscribes to `/ROBOT/wheel_odom` (configurable) and `/ROBOT/move_base/goal`.
- Publishes `/ROBOT/cmd_vel`, `/ROBOT/nav_status` and `/ROBOT/nav_path`.
- Exposes `/ROBOT/cancel_navigation` (`std_srvs/Trigger`), clearing the current goal.

## Parameters

All tunables live under `config/navigation.yaml` and are loaded for every robot through the launch file.
Key entries:

- `input_topic`/`output_topic`: remap odom/goal/cmd_vel topics per namespace.
- `controller`: gains, saturation and tolerances for the proportional controller.
- `align_final_heading`: align robot yaw with RViz goal after reaching the position.

## Planning APF (GMM 宏观规划 + APF 微观路径)

多机协同规划：给定目标 GMM 分布后，生成 GMM 宏观转化轨迹，再根据宏观轨迹利用 APF 算法计算每个无人机的微观路径。

### 流程

1. **目标 GMM**：外部发布 `apf_goal` (navigation_msgs/GMM)
2. **GMM 宏观转化轨迹**：`PlanningAPFProcess` 中 SLP 优化 + GMM 插值
3. **APF 微观路径**：每步调用 APF 计算各无人机轨迹，发布 `/{robot}/apf_trajectory`

### GMM 目标发布节点（可选）

`gmm_goal_publisher_node` 用于测试：按参数发布 GMM 到 `apf_goal`。

- **参数**：`config/gmm_goal_publisher.yaml` 中配置 `means`、`covariances`、`weights`、`publish_rate`
- **启动**：`ros2 run rover3d_navigation gmm_goal_publisher_node.py` 或 `use_gmm_publisher:=true` 随 launch 启动
- **服务**：`/publish_gmm_goal`（std_srvs/Trigger）可触发单次发布

### Planning APF Topics

- Subscribes: `/{robot}/gt/odom` (or `wheel_odom`), `apf_goal` (`navigation_msgs/GMM`)
- Publishes: `/{robot}/apf_trajectory` (`nav_msgs/Path`)
- Uses: `esdf/query` service (optional, for obstacle avoidance)

### Planning APF Launch

```bash
source install/setup.bash
ros2 launch rover3d_navigation planning_apf.launch.py
```

**参数说明：**
- `robots`: 机器人命名空间，逗号分隔，默认 `bot1,bot2,bot3`
- `config_file`: YAML 配置文件路径，默认 `planning_apf.yaml`
- `use_esdf`: 是否启动 ESDF 建图节点（`true`/`false`），默认 `false`（需单独运行 esdf_map 时）
- `use_mpc`: 是否为每机器人启动 MPC 控制器（`true`/`false`），默认 `true`
**示例：**
```bash
# 仅规划节点，ESDF 与 MPC 单独运行
ros2 launch rover3d_navigation planning_apf.launch.py use_esdf:=false use_mpc:=false

# 双机器人 + 附带 ESDF
ros2 launch rover3d_navigation planning_apf.launch.py robots:=bot1,bot2 use_esdf:=true

# 附带 GMM 目标发布（用于无外部目标源时的测试）
ros2 launch rover3d_navigation planning_apf.launch.py use_gmm_publisher:=true
```

Requires: `navigation_msgs`, `esdf_map`, `python3-numpy`, `python3-scipy`, `python3-sklearn`.

**NumPy/SciPy 兼容性**：若出现 `numpy.dtype size changed` 或二进制不兼容错误，执行：
```bash
pip install -r /path/to/fishbot_multirobot_sim/requirements.txt
```
或单独固定版本：`pip install "numpy<2.0"`

## Usage

```bash
source /fishbot_ws/install/setup.bash
ros2 launch rover3d_navigation navigation.launch.py robots:=bot1,bot2
```
