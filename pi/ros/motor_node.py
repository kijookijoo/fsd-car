from __future__ import annotations

from pathlib import Path
import json
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from ros_config import load_ros_config

ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = ROOT / "Code" / "Server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

try:
    from motor import Ordinary_Car
except Exception:
    Ordinary_Car = None


class MotorNode(Node):
    def __init__(self):
        super().__init__("motor_node")
        config = load_ros_config().get("motor")

        self.command_topic = config.get("command_topic")
        self.max_pwm = config.get("max_pwm")
        self.steering_scale = config.get("steering_scale")
        self.throttle_scale = config.get("throttle_scale")

        self.subscription = self.create_subscription(
            String,
            self.command_topic,
            self.on_command,
            10
        )
        self.motor = Ordinary_Car() if Ordinary_Car is not None else None
        if self.motor is None:
            self.get_logger().warning("motor driver unavailable")

    def _mix(self, steering: float, throttle: float):
        steering = max(-1.0, min(1.0, steering * self.steering_scale))
        throttle = max(-1.0, min(1.0, throttle * self.throttle_scale))

        left = throttle + steering
        right = throttle - steering
        scale = max(1.0, abs(left), abs(right))
        left /= scale
        right /= scale

        left_pwm = int(left * self.max_pwm)
        right_pwm = int(right * self.max_pwm)
        return left_pwm, left_pwm, right_pwm, right_pwm

    def on_command(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning("received invalid command payload")
            return

        steering = float(payload.get("steering", 0.0))
        throttle = float(payload.get("throttle", 0.0))
        wheel_pwm = self._mix(steering, throttle)

        if self.motor is not None:
            self.motor.set_motor_model(*wheel_pwm)
        else:
            self.get_logger().info(
                f"steering={steering:.3f} throttle={throttle:.3f} wheels={wheel_pwm}"
            )

    def destroy_node(self):
        if self.motor is not None:
            self.motor.close()
            self.motor = None
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MotorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
