from __future__ import annotations

from pathlib import Path
import json
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = ROOT / "Code" / "Server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from motor import Ordinary_Car
from ros_config import load_ros_config



class MotorNode(Node):
    def __init__(self):
        super().__init__("motor_node")
        config = load_ros_config()["motor"]

        self.command_topic = config["command_topic"]
        self.max_pwm = config["max_pwm"]
        self.steering_scale = config["steering_scale"]
        self.throttle_scale = config["throttle_scale"]

        self.subscription = self.create_subscription(
            String,
            self.command_topic,
            self.on_command,
            10
        )
        self.motor = Ordinary_Car()

    def compute_wheel_pwm(self, steering: float, throttle: float):
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

        steering = payload["steering"]
        throttle = payload["throttle"]
        left_front, left_rear, right_front, right_rear = self.compute_wheel_pwm(steering, throttle)
        
        if self.motor is not None:
            self.motor.set_motor_model(left_front, left_rear, right_front, right_rear)
        else:
            self.get_logger().info(
                f"steering={steering:.3f} throttle={throttle:.3f} wheels={[left_front, left_rear, right_front, right_rear]}"
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
