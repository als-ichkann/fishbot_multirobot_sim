#!/usr/bin/env python3

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class _Goal:
    x: float
    y: float
    yaw: Optional[float]
    frame_id: str
    raw_msg: PoseStamped


class NavigatorNode(Node):
    """Minimal per-robot navigator fed with odometry and RViz goals."""

    def __init__(self) -> None:
        super().__init__('navigator')
        self.robot_name = self.declare_parameter('robot_name', value='').value or 'fishbot'
        self.declare_parameter('input_topic.odom_topic', 'wheel_odom')
        self.declare_parameter('input_topic.goal_topic', 'move_base/goal')
        self.declare_parameter('output_topic.cmd_vel_topic', 'cmd_vel')
        self.declare_parameter('output_topic.nav_status_topic', 'nav_status')
        self.declare_parameter('output_topic.nav_path_topic', 'nav_path')
        self.declare_parameter('path_frame', 'odom')
        self.declare_parameter('align_final_heading', True)
        self.declare_parameter('controller.control_rate', 20.0)
        self.declare_parameter('controller.max_linear_speed', 0.8)
        self.declare_parameter('controller.max_angular_speed', 1.8)
        self.declare_parameter('controller.linear_gain', 1.2)
        self.declare_parameter('controller.angular_gain', 2.5)
        self.declare_parameter('controller.slowdown_radius', 0.5)
        self.declare_parameter('controller.arrival_tolerance', 0.1)
        self.declare_parameter('controller.yaw_tolerance', 0.05)

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.odom_sub = self.create_subscription(
            Odometry,
            self._param('input_topic.odom_topic'),
            self._on_odom,
            qos,
        )
        self.goal_sub = self.create_subscription(
            PoseStamped,
            self._param('input_topic.goal_topic'),
            self._on_goal,
            qos,
        )
        self.cmd_pub = self.create_publisher(
            Twist,
            self._param('output_topic.cmd_vel_topic'),
            qos,
        )
        self.status_pub = self.create_publisher(
            String,
            self._param('output_topic.nav_status_topic'),
            qos,
        )
        self.path_pub = self.create_publisher(
            Path,
            self._param('output_topic.nav_path_topic'),
            qos,
        )
        self.cancel_srv = self.create_service(Trigger, 'cancel_navigation', self._on_cancel)

        rate_hz = max(1.0, float(self.get_parameter('controller.control_rate').value))
        self.timer = self.create_timer(1.0 / rate_hz, self._on_timer)

        self._current_goal: Optional[_Goal] = None
        self._latest_odom: Optional[Odometry] = None
        self._nav_phase: str = 'uninitialized'
        self._path_frame = self._param('path_frame')
        self._align_heading = bool(self.get_parameter('align_final_heading').value)
        self._path_active = False
        self._arrival_tolerance = float(self.get_parameter('controller.arrival_tolerance').value)
        self._yaw_tolerance = float(self.get_parameter('controller.yaw_tolerance').value)
        self._slowdown_radius = float(self.get_parameter('controller.slowdown_radius').value)
        self._max_linear = float(self.get_parameter('controller.max_linear_speed').value)
        self._max_angular = float(self.get_parameter('controller.max_angular_speed').value)
        self._linear_gain = float(self.get_parameter('controller.linear_gain').value)
        self._angular_gain = float(self.get_parameter('controller.angular_gain').value)
        self._publish_state('idle')

    # --------------------------------------------------------------------- #
    def _param(self, name: str) -> str:
        return str(self.get_parameter(name).value)

    def _publish_state(self, state: str) -> None:
        if state == self._nav_phase:
            return
        self._nav_phase = state
        msg = String()
        msg.data = state
        self.status_pub.publish(msg)

    def _on_odom(self, msg: Odometry) -> None:
        self._latest_odom = msg

    def _on_goal(self, msg: PoseStamped) -> None:
        yaw = None
        o = msg.pose.orientation
        if abs(o.x) + abs(o.y) + abs(o.z) + abs(o.w) > 1e-6:
            yaw = _yaw_from_quaternion(o.x, o.y, o.z, o.w)
        self._current_goal = _Goal(
            x=msg.pose.position.x,
            y=msg.pose.position.y,
            yaw=yaw,
            frame_id=msg.header.frame_id or self._path_frame,
            raw_msg=msg,
        )
        self._publish_state('navigating')
        self.get_logger().info(
            f"[{self.robot_name}] New goal ({self._current_goal.x:.2f}, {self._current_goal.y:.2f})"
        )

    def _on_cancel(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        self._cancel_goal('canceled')
        response.success = True
        response.message = 'Navigation canceled'
        return response

    def _cancel_goal(self, state_msg: str) -> None:
        if self._current_goal is None:
            return
        self._current_goal = None
        self._publish_state(state_msg)
        self._stop_robot()
        self._publish_empty_path()

    def _stop_robot(self) -> None:
        twist = Twist()
        self.cmd_pub.publish(twist)

    def _publish_empty_path(self) -> None:
        if not self._path_active:
            return
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self._path_frame
        self.path_pub.publish(path)
        self._path_active = False

    def _on_timer(self) -> None:
        if self._current_goal is None:
            if self._nav_phase != 'idle':
                self._publish_state('idle')
            self._stop_robot()
            self._publish_empty_path()
            return

        if self._latest_odom is None:
            self._publish_state('waiting_for_odom')
            self._stop_robot()
            return

        state = self._drive_towards_goal()
        self._publish_path()
        if state == 'goal_reached':
            self._publish_state(state)
            self._current_goal = None
            self._publish_empty_path()
        else:
            self._publish_state(state)

    def _drive_towards_goal(self) -> str:
        assert self._current_goal is not None
        assert self._latest_odom is not None
        robot_pose = self._latest_odom.pose.pose
        rx = robot_pose.position.x
        ry = robot_pose.position.y
        current_yaw = _yaw_from_quaternion(
            robot_pose.orientation.x,
            robot_pose.orientation.y,
            robot_pose.orientation.z,
            robot_pose.orientation.w,
        )
        dx = self._current_goal.x - rx
        dy = self._current_goal.y - ry
        distance = math.hypot(dx, dy)

        if distance <= self._arrival_tolerance:
            if self._align_heading and self._current_goal.yaw is not None:
                heading_error = _normalize_angle(self._current_goal.yaw - current_yaw)
                if abs(heading_error) > self._yaw_tolerance:
                    angular = _clamp(self._angular_gain * heading_error, self._max_angular)
                    self._publish_cmd(0.0, angular)
                    return 'aligning'
            self._stop_robot()
            return 'goal_reached'

        angle_to_goal = math.atan2(dy, dx)
        heading_error = _normalize_angle(angle_to_goal - current_yaw)

        linear = self._linear_gain * distance * max(0.0, math.cos(heading_error))
        if distance < self._slowdown_radius:
            linear *= distance / max(self._slowdown_radius, 1e-3)
        angular = self._angular_gain * heading_error

        linear = _clamp(linear, self._max_linear)
        angular = _clamp(angular, self._max_angular)
        self._publish_cmd(linear, angular)
        return 'navigating'

    def _publish_cmd(self, linear: float, angular: float) -> None:
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.cmd_pub.publish(twist)

    def _publish_path(self) -> None:
        if self._current_goal is None or self._latest_odom is None:
            self._publish_empty_path()
            return
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self._path_frame
        start = PoseStamped()
        start.header = path.header
        start.pose = self._latest_odom.pose.pose
        goal = PoseStamped()
        goal.header.frame_id = self._current_goal.frame_id
        goal.header.stamp = path.header.stamp
        goal.pose = self._current_goal.raw_msg.pose
        path.poses = [start, goal]
        self.path_pub.publish(path)
        self._path_active = True


def main() -> None:
    rclpy.init()
    node = NavigatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
