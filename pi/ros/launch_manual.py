from __future__ import annotations

import sys

from launch import LaunchDescription
from launch.actions import ExecuteProcess

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROS_DIR = Path(__file__).resolve().parent


def generate_launch_description():
    return LaunchDescription(
        [
            ExecuteProcess(
                cmd=[sys.executable, str(ROS_DIR / "camera_node.py")],
                cwd=str(ROS_DIR),
                output="screen",
            ),
            ExecuteProcess(
                cmd=[sys.executable, str(ROS_DIR / "teleop_node.py")],
                cwd=str(ROS_DIR),
                output="screen",
            ),
            ExecuteProcess(
                cmd=[sys.executable, str(ROS_DIR / "logger_node.py")],
                cwd=str(ROS_DIR),
                output="screen",
            ),
            ExecuteProcess(
                cmd=[sys.executable, str(ROS_DIR / "motor_node.py")],
                cwd=str(ROS_DIR),
                output="screen",
            ),
        ]
    )
