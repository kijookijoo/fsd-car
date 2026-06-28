from __future__ import annotations

import csv
import os
import sys
import termios
import tty
from dataclasses import dataclass
from pathlib import Path
from select import select
from time import time, sleep

import cv2


BASE_DATA_DIR = Path("./data")
CAPTURE_INTERVAL_SEC = 0.2
CAMERA_INDEX = 0


@dataclass
class ControlState:
    steering: float = 0.0
    throttle: float = 0.0


class Recorder:
    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
        self.images_dir = self.session_dir / "images"
        self.csv_path = self.session_dir / "labels.csv"
        self.recording = False
        self.counter = 0
        self.last_capture_time = 0.0

        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_csv_header()

    def _ensure_csv_header(self):
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["image", "steering", "throttle", "timestamp"])

    def save_sample(self, frame, controls: ControlState):
        self.counter += 1
        filename = f"{self.counter:06d}.jpg"
        image_path = self.images_dir / filename

        cv2.imwrite(str(image_path), frame)

        with self.csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([filename, controls.steering, controls.throttle, time()])

    def maybe_save_sample(self, frame, controls: ControlState):
        now = time()
        if now - self.last_capture_time >= CAPTURE_INTERVAL_SEC:
            self.last_capture_time = now
            self.save_sample(frame, controls)

    def toggle_recording(self):
        self.recording = not self.recording
        return self.recording


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def next_session_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = []
    for path in base_dir.iterdir():
        if path.is_dir() and path.name.startswith("session_"):
            suffix = path.name.removeprefix("session_")
            if suffix.isdigit():
                existing.append(int(suffix))
    next_index = (max(existing) + 1) if existing else 1
    return base_dir / f"session_{next_index:03d}"


def get_key(timeout_sec: float = 0.05):
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        readable, _, _ = select([sys.stdin], [], [], timeout_sec)
        if readable:
            return sys.stdin.read(1)
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def stream_camera(camera_index: int = CAMERA_INDEX):
    session_dir = next_session_dir(BASE_DATA_DIR)
    recorder = Recorder(session_dir=session_dir)
    controls = ControlState()

    print(f"session_dir={session_dir}")
    print("Controls:")
    print("  c   = capture one frame")
    print("  r   = toggle recording")
    print("  w/s = throttle up/down")
    print("  a/d = steering left/right")
    print("  x   = reset controls")
    print("  q   = quit")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 642)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame = cv2.flip(frame, 1)

            key = get_key()
            if key == "q":
                break
            elif key == "r":
                print(f"recording={recorder.toggle_recording()}")
            elif key == "c":
                recorder.save_sample(frame, controls)
                print("saved one frame")
            elif key == "x":
                controls.steering = 0.0
                controls.throttle = 0.0
                print("controls reset")
            elif key == "w":
                controls.throttle = clamp(controls.throttle + 0.05, -1.0, 1.0)
                print(f"throttle={controls.throttle:.2f}")
            elif key == "s":
                controls.throttle = clamp(controls.throttle - 0.05, -1.0, 1.0)
                print(f"throttle={controls.throttle:.2f}")
            elif key == "a":
                controls.steering = clamp(controls.steering - 0.05, -1.0, 1.0)
                print(f"steering={controls.steering:.2f}")
            elif key == "d":
                controls.steering = clamp(controls.steering + 0.05, -1.0, 1.0)
                print(f"steering={controls.steering:.2f}")

            if recorder.recording:
                recorder.maybe_save_sample(frame, controls)

            sleep(0.01)

    finally:
        cap.release()


if __name__ == "__main__":
    stream_camera()
