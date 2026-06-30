from __future__ import annotations

import csv
import json
from collections import deque
from pathlib import Path

import cv2

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from ros_config import load_ros_config, resolve_path


class LoggerNode(Node):
    def __init__(self):
        super().__init__("logger_node")
        config = load_ros_config().get("logger")

        self.frame_topic = config.get("frame_topic")
        self.command_topic = config.get("command_topic")
        self.log_path = resolve_path(config.get("log_path"))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.log_path.parent / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.bridge = CvBridge()
        self.frames = {}
        self.frame_order = deque()
        self.pending_commands = {}
        self.pending_order = deque()
        self.max_buffered_frames = 200

        self.frame_subscription = self.create_subscription(
            Image, self.frame_topic, self.on_frame, 10
        )
        self.command_subscription = self.create_subscription(
            String, self.command_topic, self.on_command, 10
        )
        self.file = self.log_path.open("a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        if self.file.tell() == 0:
            self.writer.writerow(["image", "frame_stamp_ns", "steering", "throttle"])
            self.file.flush()

    def on_frame(self, msg: Image) -> None:
        frame_stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.frames[frame_stamp_ns] = frame
        self.frame_order.append(frame_stamp_ns)
        self._trim_frame_buffer()
        self._flush_matching_sample(frame_stamp_ns)

    def _trim_frame_buffer(self) -> None:
        while len(self.frame_order) > self.max_buffered_frames:
            old_stamp = self.frame_order.popleft()
            self.frames.pop(old_stamp, None)
        while len(self.pending_order) > self.max_buffered_frames:
            old_stamp = self.pending_order.popleft()
            self.pending_commands.pop(old_stamp, None)

    def _flush_matching_sample(self, frame_stamp_ns: int) -> None:
        payload = self.pending_commands.pop(frame_stamp_ns, None)
        if payload is None:
            return
        try:
            self.pending_order.remove(frame_stamp_ns)
        except ValueError:
            pass
        self._write_sample(frame_stamp_ns, payload)

    def _write_sample(self, frame_stamp_ns: int, payload: dict) -> None:
        frame = self.frames.pop(frame_stamp_ns, None)
        if frame is None:
            self.get_logger().warning(
                f"missing frame for stamp_ns={frame_stamp_ns}, skipping sample"
            )
            return
        try:
            self.frame_order.remove(frame_stamp_ns)
        except ValueError:
            pass

        steering = float(payload.get("steering", 0.0))
        throttle = float(payload.get("throttle", 0.0))

        image_name = f"{frame_stamp_ns}.jpg"
        image_path = self.images_dir / image_name
        if not cv2.imwrite(str(image_path), frame):
            raise RuntimeError(f"failed to write image: {image_path}")

        self.writer.writerow(
            [image_name, frame_stamp_ns, steering, throttle]
        )
        self.file.flush()

    def on_command(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning("received invalid command payload")
            return

        frame_stamp_ns = int(payload.get("frame_stamp_ns", -1))
        if frame_stamp_ns in self.frames:
            self._write_sample(frame_stamp_ns, payload)
        else:
            self.pending_commands[frame_stamp_ns] = payload
            self.pending_order.append(frame_stamp_ns)
            self._trim_frame_buffer()

    def destroy_node(self):
        if getattr(self, "file", None):
            self.file.close()
            self.file = None
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
