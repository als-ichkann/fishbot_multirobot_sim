#!/usr/bin/env python3
"""
Planning APF 节点：采用 PlanningAPFProcess 逻辑，GMM 插值 + APF 生成轨迹并发布。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import rclpy
from navigation_msgs.msg import GMM
from rover3d_navigation.ROVER_3D import PlanningAPFProcess
from rover3d_navigation.esdf_adapter import EsdfMapAdapter
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy


class PlanningAPFNode(Node):
    """
    规划 APF 节点：订阅 GMM 目标与多机 odom，运行 PlanningAPFProcess，
    发布每机器人轨迹 Path。
    """

    def __init__(self) -> None:
        super().__init__("planning_apf_node")

        self.declare_parameter("robot_names", ["bot1", "bot2"])
        self.declare_parameter("odom_suffix", "gt/odom")
        self.declare_parameter("goal_topic", "apf_goal")
        self.declare_parameter("trajectory_suffix", "apf_trajectory")
        self.declare_parameter("control_rate", 5.0)
        self.declare_parameter("max_apf_try", 10)
        self.declare_parameter("gmm_interp_steps", 5)
        self.declare_parameter("use_gmm_trajectory_slp", True)
        self.declare_parameter("esdf_service", "esdf/query")
        self.declare_parameter("esdf_frame_id", "map_origin")
        self.declare_parameter("map_origin_x", -5.0)
        self.declare_parameter("map_origin_y", -7.5)
        self.declare_parameter("map_origin_z", 0.0)
        self.declare_parameter("map_size_x", 22.0)
        self.declare_parameter("map_size_y", 17.0)
        self.declare_parameter("map_size_z", 6.0)
        self.declare_parameter("esdf_resolution", 0.15)
        self.declare_parameter("grid_step", 2.0)
        self.declare_parameter("config_dir", "")

        robot_names = self.get_parameter("robot_names").value
        if isinstance(robot_names, str):
            robot_names = [n.strip() for n in robot_names.split(",") if n.strip()]
        self._robot_names = list(robot_names)
        self._odom_suffix = str(self.get_parameter("odom_suffix").value)
        self._goal_topic = str(self.get_parameter("goal_topic").value)
        self._traj_suffix = str(self.get_parameter("trajectory_suffix").value)
        self._control_rate = float(self.get_parameter("control_rate").value)
        self._max_apf_try = int(self.get_parameter("max_apf_try").value)
        self._gmm_interp_steps = int(self.get_parameter("gmm_interp_steps").value)
        self._use_gmm_trajectory_slp = bool(
            self.get_parameter("use_gmm_trajectory_slp").value
        )
        config_dir_param = str(self.get_parameter("config_dir").value)
        if config_dir_param:
            self._config_dir = config_dir_param
        else:
            try:
                from ament_index_python.packages import get_package_share_directory
                pkg_share = get_package_share_directory("rover3d_navigation")
                self._config_dir = os.path.join(pkg_share, "config")
            except Exception:
                self._config_dir = ""

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self._odom_cache: Dict[str, Optional[Odometry]] = {
            name: None for name in self._robot_names
        }
        self._gmm_goal: Optional[tuple] = None
        self._planning_process: Optional[PlanningAPFProcess] = None
        self._log_no_odom_done = False
        self._log_no_traj_done = False
        self._last_traj_log_time = 0.0
        self._first_publish_done = False

        for name in self._robot_names:
            self.create_subscription(
                Odometry,
                f"{name}/{self._odom_suffix}",
                lambda msg, n=name: self._cb_odom(n, msg),
                qos,
            )
        self.create_subscription(GMM, self._goal_topic, self._cb_gmm, 10)

        self._path_pubs: Dict[str, object] = {}
        for name in self._robot_names:
            self._path_pubs[name] = self.create_publisher(
                Path, f"{name}/{self._traj_suffix}", 10
            )

        self._esdf = EsdfMapAdapter(
            self,
            service_name=self.get_parameter("esdf_service").value,
            frame_id=self.get_parameter("esdf_frame_id").value,
            map_origin_x=float(self.get_parameter("map_origin_x").value),
            map_origin_y=float(self.get_parameter("map_origin_y").value),
            map_origin_z=float(self.get_parameter("map_origin_z").value),
            map_size_x=float(self.get_parameter("map_size_x").value),
            map_size_y=float(self.get_parameter("map_size_y").value),
            map_size_z=float(self.get_parameter("map_size_z").value),
            resolution=float(self.get_parameter("esdf_resolution").value),
        )

        self.create_timer(1.0 / self._control_rate, self._control_loop)
        self.get_logger().info(
            f"Planning APF: robots={self._robot_names}, goal={self._goal_topic}, "
            f"gmm_interp_steps={self._gmm_interp_steps}"
        )

    def _cb_odom(self, name: str, msg: Odometry) -> None:
        self._odom_cache[name] = msg

    def _cb_gmm(self, msg: GMM) -> None:
        n = len(msg.means)
        if n == 0:
            self._gmm_goal = None
            self._planning_process = None
            return
        if len(msg.covariances) < n * 9 or len(msg.weights) < n:
            self.get_logger().warn(
                "GMM msg invalid: covariances or weights length mismatch"
            )
            return
        means = [(m.x, m.y, m.z) for m in msg.means]
        covs = []
        for i in range(n):
            start = i * 9
            end = start + 9
            c = np.array(msg.covariances[start:end]).reshape(3, 3)
            covs.append(c)
        weights = list(msg.weights[:n])
        self._gmm_goal = (means, covs, weights)
        self._planning_process = None
        self.get_logger().info(f"New GMM goal: {n} components")

    def _get_robots_positions(self) -> Optional[np.ndarray]:
        positions = []
        missing = []
        for name in self._robot_names:
            odom = self._odom_cache.get(name)
            if odom is None:
                missing.append(name)
            else:
                p = odom.pose.pose.position
                positions.append([p.x, p.y, p.z])
        if missing:
            if not self._log_no_odom_done:
                self.get_logger().warn(
                    f"Waiting for odom: missing {missing}. "
                    f"Run sim and ensure /<robot>/{self._odom_suffix} is published for each robot."
                )
                self._log_no_odom_done = True
            return None
        self._log_no_odom_done = False
        return np.array(positions, dtype=float)

    def _control_loop(self) -> None:
        if self._gmm_goal is None:
            self._publish_empty_paths()
            return
        robots_positions = self._get_robots_positions()
        if robots_positions is None or len(robots_positions) == 0:
            return
        means, covs, weights = self._gmm_goal
        if len(means) == 0:
            self._publish_empty_paths()
            return

        if self._planning_process is None:
            self.get_logger().info(
                "Creating PlanningAPFProcess"
            )
            xa = float(self.get_parameter("map_origin_x").value)
            ya = float(self.get_parameter("map_origin_y").value)
            za = float(self.get_parameter("map_origin_z").value)
            xb = xa + float(self.get_parameter("map_size_x").value)
            yb = ya + float(self.get_parameter("map_size_y").value)
            zb = za + float(self.get_parameter("map_size_z").value)
            grid_step = float(self.get_parameter("grid_step").value)
            self._planning_process = PlanningAPFProcess(
                num_robots=len(self._robot_names),
                esdf_map=self._esdf,
                xa=xa,
                xb=xb,
                ya=ya,
                yb=yb,
                za=za,
                zb=zb,
                goal_means=means,
                goal_covs=covs,
                goal_weights=weights,
                gmm_interp_steps=self._gmm_interp_steps,
                max_apf_try=self._max_apf_try,
                use_gmm_trajectory_slp=self._use_gmm_trajectory_slp,
                grid_step=grid_step,
                config_dir=self._config_dir if self._config_dir else None,
            )

        try:
            result = self._planning_process.run_one_cycle(robots_positions)
        except Exception as e:
            import traceback
            self.get_logger().error(f"PlanningAPFProcess.run_one_cycle failed: {e}")
            self.get_logger().error(traceback.format_exc())
            return
        if result is None:
            if not self._log_no_traj_done:
                self.get_logger().info(
                    "run_one_cycle returned None (GMM optimizing or done). "
                    "Trajectories will publish when ready."
                )
                self._log_no_traj_done = True
            if self._planning_process.stop_flag:
                self._publish_empty_paths()
            return
        self._log_no_traj_done = False
        trajectories, _ = result
        path_frame = "map_origin"
        stamp = self.get_clock().now().to_msg()
        for i, name in enumerate(self._robot_names):
            path = Path()
            path.header.stamp = stamp
            path.header.frame_id = path_frame
            if i < len(trajectories):
                for pt in trajectories[i]:
                    ps = PoseStamped()
                    ps.header = path.header
                    ps.pose.position.x = float(pt[0])
                    ps.pose.position.y = float(pt[1])
                    ps.pose.position.z = float(pt[2])
                    ps.pose.orientation.w = 1.0
                    path.poses.append(ps)
            if len(path.poses) == 0:
                pos = robots_positions[i]
                ps = PoseStamped()
                ps.header = path.header
                ps.pose.position.x = float(pos[0])
                ps.pose.position.y = float(pos[1])
                ps.pose.position.z = float(pos[2])
                ps.pose.orientation.w = 1.0
                path.poses.append(ps)
            self._path_pubs[name].publish(path)
        num_pts = sum(len(trajectories[i]) for i in range(min(len(trajectories), len(self._robot_names))))
        if not self._first_publish_done:
            self.get_logger().info(
                f"Published first trajectories: {num_pts} pose(s), frame={path_frame}, "
                f"topics={[f'{r}/{self._traj_suffix}' for r in self._robot_names]}"
            )
            self._first_publish_done = True
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self._last_traj_log_time > 2.0:
            self.get_logger().info(
                f"Published trajectories: {num_pts} pose(s) for {len(self._robot_names)} robots"
            )
            self._last_traj_log_time = now

    def _publish_empty_paths(self) -> None:
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = "map_origin"
        for pub in self._path_pubs.values():
            pub.publish(path)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PlanningAPFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
