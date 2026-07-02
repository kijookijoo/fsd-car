from __future__ import annotations

import io
import threading
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from flask import Flask, Response
from rclpy.node import Node
from sensor_msgs.msg import Image

from ros_config import load_ros_config


class WebServerNode(Node):
    def __init__(self):
        super().__init__("web_server_node")
        config = load_ros_config()["camera"]

        self.frame_topic = config["topic"]
        self.bridge = CvBridge()
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        self.subscription = self.create_subscription(
            Image,
            self.frame_topic,
            self.on_frame,
            10
        )

        self.get_logger().info(f"Subscribed to {self.frame_topic}")

    def on_frame(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self.frame_lock:
            self.latest_frame = frame

    def get_frame_jpeg(self) -> bytes:
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            _, buffer = cv2.imencode(".jpg", self.latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return buffer.tobytes()


def create_flask_app(ros_node: WebServerNode) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>FSD Car Camera</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {
                    font-family: Arial, sans-serif;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    min-height: 100vh;
                    margin: 0;
                    background-color: #222;
                    color: #fff;
                }
                .container {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    gap: 20px;
                }
                h1 {
                    margin: 0;
                }
                img {
                    max-width: 90vw;
                    max-height: 80vh;
                    border: 2px solid #444;
                    border-radius: 4px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>FSD Car Camera Stream</h1>
                <img id="stream" alt="Camera stream loading..." />
            </div>
            <script>
                const img = document.getElementById('stream');
                function updateFrame() {
                    img.src = '/frame.jpg?' + new Date().getTime();
                }
                updateFrame();
                setInterval(updateFrame, 100);
            </script>
        </body>
        </html>
        """

    @app.route("/frame.jpg")
    def frame_jpg():
        frame_data = ros_node.get_frame_jpeg()
        if frame_data is None:
            return "No frame available", 503
        return Response(frame_data, mimetype="image/jpeg")

    @app.route("/stream.mjpg")
    def stream_mjpg():
        def generate():
            while True:
                frame_data = ros_node.get_frame_jpeg()
                if frame_data is not None:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(frame_data)).encode() + b"\r\n\r\n"
                        + frame_data + b"\r\n"
                    )
                rclpy.spin_once(ros_node, timeout_sec=0.01)

        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    return app


def main(args=None):
    rclpy.init(args=args)
    node = WebServerNode()
    app = create_flask_app(node)

    def ros_spin():
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass

    ros_thread = threading.Thread(target=ros_spin, daemon=True)
    ros_thread.start()

    node.get_logger().info("Starting web server on http://0.0.0.0:5000")
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
