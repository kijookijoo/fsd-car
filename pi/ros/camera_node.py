from __future__ import annotations

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from ros_config import load_ros_config


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")
        config = load_ros_config().get("camera", {})

        self.publisher = self.create_publisher(
            Image, 
            str(config.get("topic")), 
            10
            )
        self.bridge = CvBridge()
        self.capture = cv2.VideoCapture(config.get("index"))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.get("width"))
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.get("height"))

        if not self.capture.isOpened():
            self.get_logger().warning(
                f"camera index {config.get('index')} could not be opened"
            )

        self.timer = self.create_timer(
            config.get("publish_hz"), 
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
        rclpy.shutdown()


if __name__ == "__main__":
    main()
