from __future__ import annotations

from importlib import util
from pathlib import Path
import sys
import json

import cv2
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from torchvision import transforms

from ros_config import load_ros_config, resolve_path

def load_cnn_class():
    spec = util.spec_from_file_location("fsd_cnn")
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load CNN module.")
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.CNN


class InferenceNode(Node):
    def __init__(self):
        super().__init__("inference_node")
        config = load_ros_config().get("inference")
        self.model_path = resolve_path(config.get("model_path"))
        self.publish_hz = config.get("publish_hz")
        self.publisher = self.create_publisher(
            String,
            config.get("command_topic"),
            10
            )
        self.subscription = self.create_subscription(
            Image, 
            config.get("frame_topic"), 
            self.on_frame, 
            10
            )   
        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.CNN = load_cnn_class()
        self.model = self._load_model()
        self.preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((66, 200)),
                transforms.ToTensor(),
            ]
        )
        self.latest_frame = None
        self.latest_frame_stamp_ns = None
        self.timer = self.create_timer(1.0 / self.publish_hz, self.publish_command)

    def _load_model(self):
        model = self.CNN(input_shape=(3, 66, 200), output_dim=2).to(self.device)
        if self.model_path.exists():
            checkpoint = torch.load(self.model_path, map_location=self.device)
            state = checkpoint.get("model_state")
            model.load_state_dict(state)
            self.get_logger().info(f"loaded model")
        else:
            self.get_logger().warning(
                f"model file {self.model_path} not found, using an untrained model"
            )
        model.eval()
        return model

    def on_frame(self, msg: Image) -> None:    
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.latest_frame_stamp_ns = (
            int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        )

    def publish_command(self) -> None:
        if self.latest_frame is None or self.latest_frame_stamp_ns is None:
            return

        frame_rgb = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB)
        image = self.preprocess(frame_rgb).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            prediction = self.model(image).squeeze(0).detach().cpu()

        command = String()
        command.data = json.dumps(
            {
                "frame_stamp_ns": self.latest_frame_stamp_ns,
                "steering": float(prediction[0]),
                "throttle": float(prediction[1]),
            }
        )
        self.publisher.publish(command)


def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
