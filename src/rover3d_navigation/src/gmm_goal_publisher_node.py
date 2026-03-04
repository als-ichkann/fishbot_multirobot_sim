#!/usr/bin/env python3
"""
GMM Goal Publisher: 发布目标 GMM 分布到 apf_goal 供 Planning APF 使用。

支持从参数或配置文件读取 GMM（means、covariances、weights），
按设定频率周期发布，或通过服务触发单次发布。
"""

from __future__ import annotations

from typing import List

import rclpy
from navigation_msgs.msg import GMM
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_srvs.srv import Trigger


class GMMGoalPublisherNode(Node):
    """
    发布 GMM 目标到 apf_goal。
    参数：
      - goal_topic: 发布话题，默认 apf_goal
      - publish_rate: 发布频率 Hz，0 表示仅通过服务触发
      - means: [[x,y,z], ...] 各高斯分量均值（以 map_origin 为参考）
      - covariances: 每个分量 9 个浮点数（3x3 行优先），或省略则用默认各向同性
      - weights: 各分量权重，需非负且可归一化
    """

    def __init__(self) -> None:
        super().__init__("gmm_goal_publisher")

        self.declare_parameter("goal_topic", "apf_goal")
        self.declare_parameter("publish_rate", 1.0)
        # means/covs/weights 从 YAML 加载（支持嵌套结构），命令行无法直接覆盖
        self.declare_parameter("means", [0.0, 0.0, 0.0])  # 单分量时 [x,y,z]
        self.declare_parameter("covariances", [])
        self.declare_parameter("weights", [1.0])
        self.declare_parameter("default_covariance_scale", 0.5)

        def _get(name, default):
            try:
                return self.get_parameter(name).value
            except Exception:
                return default

        goal_topic = _get("goal_topic", "apf_goal")
        publish_rate = float(_get("publish_rate", 1.0))
        means_raw = _get("means", [0.0, 0.0, 0.0])
        covs_raw = _get("covariances", [])
        weights_raw = _get("weights", [1.0])
        cov_scale = float(_get("default_covariance_scale", 0.5))

        self._means = self._parse_means(means_raw)
        self._covs, self._weights = self._parse_covs_weights(
            covs_raw, weights_raw, len(self._means), cov_scale
        )

        self._pub = self.create_publisher(GMM, goal_topic, 10)
        self._trigger_srv = self.create_service(
            Trigger, "publish_gmm_goal", self._on_trigger
        )

        if publish_rate > 0:
            self.create_timer(1.0 / publish_rate, self._publish_gmm)
            self.get_logger().info(
                f"Publishing GMM to {goal_topic} at {publish_rate} Hz, "
                f"{len(self._means)} component(s)"
            )
        else:
            self.get_logger().info(
                f"GMM publisher ready. Call ~/publish_gmm_goal to publish once."
            )

    def _parse_means(self, raw) -> List[List[float]]:
        means = []
        if not raw:
            return [[0.0, 0.0, 0.0]]
        # 支持单分量 [x,y,z] 或多分量 [[x,y,z], ...]
        if isinstance(raw[0], (int, float)):
            raw = [raw]
        for m in raw:
            if isinstance(m, (list, tuple)) and len(m) >= 3:
                means.append([float(m[0]), float(m[1]), float(m[2])])
            elif isinstance(m, dict) and "x" in m and "y" in m and "z" in m:
                means.append([float(m["x"]), float(m["y"]), float(m["z"])])
            else:
                self.get_logger().warn(f"Skipping invalid mean: {m}")
        return means if means else [[0.0, 0.0, 0.0]]

    def _parse_covs_weights(
        self, covs_raw: list, weights_raw: list, n: int, cov_scale: float
    ) -> tuple:
        weights = []
        for w in weights_raw[:n]:
            weights.append(float(w))
        while len(weights) < n:
            weights.append(1.0)
        weights = weights[:n]
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]

        covs = []
        if len(covs_raw) >= n * 9:
            for i in range(n):
                start = i * 9
                c = list(covs_raw[start : start + 9])
                if len(c) == 9:
                    covs.append(c)
            if len(covs) == n:
                return covs, weights

        for _ in range(n):
            covs.append(
                [
                    float(cov_scale), 0.0, 0.0,
                    0.0, float(cov_scale), 0.0,
                    0.0, 0.0, float(cov_scale),
                ]
            )
        return covs, weights

    def _build_gmm_msg(self) -> GMM:
        msg = GMM()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map_origin"
        for m in self._means:
            p = Point()
            p.x, p.y, p.z = float(m[0]), float(m[1]), float(m[2])
            msg.means.append(p)
        msg.covariances = [float(v) for c in self._covs for v in c]
        msg.weights = [float(w) for w in self._weights]
        return msg

    def _publish_gmm(self) -> None:
        self._pub.publish(self._build_gmm_msg())

    def _on_trigger(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        self._publish_gmm()
        response.success = True
        response.message = f"Published GMM with {len(self._means)} component(s)"
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GMMGoalPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
