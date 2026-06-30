from __future__ import annotations

import argparse
import sys
from pathlib import Path

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.launch_service import LaunchService


ROOT = Path(__file__).resolve().parents[1]
ROS_DIR = Path(__file__).resolve().parent


def run_launch_file(launch_file: Path) -> int:
    source = PythonLaunchDescriptionSource(str(launch_file))
    service = LaunchService()
    service.include_launch_description(IncludeLaunchDescription(source))
    return service.run()


def launch_manual() -> int:
    return run_launch_file(ROS_DIR / "launch_manual.py")


def launch_inference() -> int:
    return run_launch_file(ROS_DIR / "launch_inference.py")


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch a recording or inference session.")
    parser.add_argument(
        "mode",
        choices=["manual", "inference"],
        help="manual = data collection, inference = self-driving runtime",
    )
    args = parser.parse_args()

    if args.mode == "manual":
        return launch_manual()
    return launch_inference()


if __name__ == "__main__":
    raise SystemExit(main())
