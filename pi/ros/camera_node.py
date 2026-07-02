from __future__ import annotations

import glob
import pathlib
import subprocess

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from ros_config import load_ros_config


def find_camera_device() -> str | None:
    """
    Find a valid camera input device by attempting to open each /dev/video* device.
    Returns the device path that successfully opens, starting with /dev/video0.
    """
    video_devices = ["/dev/video0"] + sorted(glob.glob("/dev/video[1-9]*"))

    for device_path in video_devices:
        if not pathlib.Path(device_path).exists():
            continue

        try:
            test_cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
            if test_cap.isOpened():
                test_cap.release()
                return device_path
            test_cap.release()
        except Exception:
            continue

    return None


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")
        config = load_ros_config()["camera"]

        self.publisher = self.create_publisher(
            Image,
            config["topic"],
            10
            )
        self.bridge = CvBridge()

        device_path = find_camera_device()
        if device_path is None:
            self.get_logger().error(
                "Could not find any camera device. "
                "Please check that a camera is connected (/dev/video0 or similar)."
            )
            self.capture = None
        else:
            self.get_logger().info(f"Opening camera device: {device_path}")
            self.capture = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, config["width"])
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config["height"])

            if not self.capture.isOpened():
                self.get_logger().error(
                    f"Camera device {device_path} could not be opened. "
                    f"Check device permissions: ls -l {device_path}"
                )

        timer_period = 1.0 / config["publish_hz"]
        self.timer = self.create_timer(
            timer_period,
            self.publish_frame
            )

    def publish_frame(self) -> None:
        if not self.capture or not self.capture.isOpened():
            return

        ok, frame = self.capture.read()
        if not ok:
            self.get_logger().warning("camera read failed")
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(msg)

    def destroy_node(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
