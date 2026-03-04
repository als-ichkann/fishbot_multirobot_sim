#!/usr/bin/env python3
"""
MPC Drone Control Launch

为每台机器人启动 MPC 轨迹跟踪控制器，订阅 apf_trajectory，发布 cmd_vel。

使用方法:
  ros2 launch mpc_control mpc_control.launch.py
  ros2 launch mpc_control mpc_control.launch.py robots:=bot1,bot2,bot3

前提：
  - planning_apf 节点已运行，并发布 /{robot}/apf_trajectory
  - 仿真/定位发布 /{robot}/gt/odom
"""
from typing import List

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _parse_robot_names(raw: str) -> List[str]:
    return [n.strip() for n in raw.split(",") if n.strip()] or ["bot1"]


def _mpc_control_setup(context, *args, **kwargs):
    robots_raw = LaunchConfiguration("robots").perform(context)
    vel_scale = LaunchConfiguration("velocity_scale", default="500.0").perform(context)
    min_speed = LaunchConfiguration("min_speed", default="0.15").perform(context)
    robot_names = _parse_robot_names(robots_raw)

    nodes = []
    for name in robot_names:
        prefix = f"/{name}"
        nodes.append(
            Node(
                package="mpc_control",
                executable="mpc_drone_control",
                name=f"mpc_drone_control_{name}",
                namespace=name,
                output="screen",
                parameters=[{"velocity_scale": float(vel_scale), "min_speed": float(min_speed)}],
                remappings=[
                    ("gt/odom", f"{prefix}/gt/odom"),
                    ("trajectory", f"{prefix}/apf_trajectory"),
                    ("cmd_vel", f"{prefix}/cmd_vel"),
                ],
            )
        )
        # Gazebo MulticopterVelocityControl 需收到 enable=true 才响应 cmd_vel
        from launch.actions import ExecuteProcess
        enable_cmd = ExecuteProcess(
            cmd=["ros2", "topic", "pub", "--once", f"{prefix}/enable", "std_msgs/msg/Bool", "{data: true}"],
            output="log",
        )
        nodes.append(TimerAction(period=2.0, actions=[enable_cmd]))

    return nodes


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument(
            "robots",
            default_value="bot1,bot2,bot3",
            description="逗号分隔的机器人命名空间，如 bot1,bot2,bot3",
        ),
        DeclareLaunchArgument(
            "velocity_scale",
            default_value="500.0",
            description="MPC 速度输出增益（MPC 输出很小，需大倍数放大）",
        ),
        DeclareLaunchArgument(
            "min_speed",
            default_value="0.15",
            description="轨迹跟踪时的最小速度 (m/s)，避免输出过小无人机不动",
        ),
        OpaqueFunction(function=_mpc_control_setup),
    ])
