from __future__ import annotations

import csv

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from ros_config import load_ros_config, resolve_path


class LoggerNode(Node):
    def __init__(self):
        super().__init__("logger_node")
        config = load_ros_config().get("logger")

        self.command_topic = config.get("command_topic")
        self.log_path = resolve_path(config.get("log_path"))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.sample_index = 0

        self.subscription = self.create_subscription(
            Float32MultiArray, self.command_topic, self.on_command, 10
        )
        self.file = self.log_path.open("a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        if self.file.tell() == 0:
            self.writer.writerow(["image", "steering", "throttle"])
            self.file.flush()

    def on_command(self, msg: Float32MultiArray) -> None:
        steering, throttle = msg.data
        self.sample_index += 1
        image_name = f"{self.sample_index:06d}.jpg"
        self.writer.writerow([image_name, steering, throttle])
        self.file.flush()

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
