from __future__ import annotations

from typing import List

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _parse_robot_names(raw: str) -> List[str]:
    robots = [name.strip() for name in raw.split(',') if name.strip()]
    return robots or ['bot1']


def _configure_nodes(context, *args, **kwargs):
    robots_value = LaunchConfiguration('robots').perform(context)
    robot_names = _parse_robot_names(robots_value)
    config_file = LaunchConfiguration('config_file')

    nodes = []
    for name in robot_names:
        nodes.append(
            Node(
                package='rover3d_navigation',
                executable='navigator.py',
                namespace=name,
                name='navigator',
                output='screen',
                parameters=[config_file, {'robot_name': name}],
            )
        )
    return nodes


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument('robots', default_value='bot1', description='Comma separated robot namespaces.'),
        DeclareLaunchArgument(
            'config_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('rover3d_navigation'),
                'config',
                'navigation.yaml',
            ]),
            description='YAML file with default navigator parameters.',
        ),
        OpaqueFunction(function=_configure_nodes),
    ])
