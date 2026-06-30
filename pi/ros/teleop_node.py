from __future__ import annotations

import json
import sys
import termios
import tty
from dataclasses import dataclass
from select import select

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from ros_config import load_ros_config


@dataclass
class TeleopState:
    linear: float = 0.0
    angular: float = 0.0


class TeleopNode(Node):
    def __init__(self):
        super().__init__("teleop_node")
        config = load_ros_config()["teleop"]

        self.command_topic = config["command_topic"]
        self.publish_hz = config["publish_hz"]
        self.linear_step = config["linear_step"]
        self.angular_step = config["angular_step"]
        self.max_linear = config["max_linear"]
        self.max_angular = config["max_angular"]
        self.zero_on_idle = config["zero_on_idle"]

        self.publisher = self.create_publisher(
            String, 
            self.command_topic, 
            10
            )
        self.state = TeleopState()
        self._stdin_fd = sys.stdin.fileno()
        self._old_term_settings = termios.tcgetattr(self._stdin_fd)
        
        # instant input without having to press Enter every command
        tty.setcbreak(self._stdin_fd)
        
        # keep moving in the current direction unless it gets an input
        self._last_published = None 
        self.timer = self.create_timer(1.0 / self.publish_hz, self.poll_and_publish)
        self.get_logger().info(
            "teleop controls: w/s linear, a/d angular, x stop, q quit"
        )

    def _read_key(self, timeout_sec: float = 0.0):
        readable, _, _ = select([sys.stdin], [], [], timeout_sec)
        if readable:
            return sys.stdin.read(1)
        return None

    def bound(self, value: float, limit: float) -> float:
        return max(-limit, min(limit, value))

    def _publish(self) -> None:
        payload = {
            "frame_stamp_ns": -1,
            "steering": float(self.state.angular),
            "throttle": float(self.state.linear),
        }
        msg = String()
        msg.data = json.dumps(payload)
        self.publisher.publish(msg)
        self._last_published = (self.state.linear, self.state.angular)

    def poll_and_publish(self) -> None:
        key = self._read_key()

        if key is not None:
            if key == "q":
                rclpy.shutdown()
                return
            if key == "x":
                self.state.linear = 0.0
                self.state.angular = 0.0
            elif key == "w":
                self.state.linear = self.bound(
                    self.state.linear + self.linear_step, self.max_linear
                )
            elif key == "s":
                self.state.linear = self.bound(
                    self.state.linear - self.linear_step, self.max_linear
                )
            elif key == "a":
                self.state.angular = self.bound(
                    self.state.angular + self.angular_step, self.max_angular
                )
            elif key == "d":
                self.state.angular = self.bound(
                    self.state.angular - self.angular_step, self.max_angular
                )

        if key is None and self.zero_on_idle:
            self.state.linear = 0.0
            self.state.angular = 0.0

        current = (self.state.linear, self.state.angular)
        if current != self._last_published:
            self._publish()

    def destroy_node(self):
        termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._old_term_settings)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
