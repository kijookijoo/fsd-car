# End-to-End Self-Driving RC Car

This project is a ROS2-based self-driving RC car built around behavioral cloning.
The system learns from human driving data, then reuses the same pipeline for autonomous inference.

## Core Idea

The car does not rely on hand-written lane rules or a large perception stack.
It learns a direct mapping from camera frames to control commands.

```text
camera frame -> model -> steering + throttle
```

## System Overview

The architecture is split into three parts:

```text
manual/teleop mode      training pipeline         inference mode
-------------------     -----------------         --------------
human drives car  ->    dataset creation   ->     camera -> model -> motor
```

The ROS2 nodes keep this flow modular:

```text
camera_node
  -> publishes frames

teleop_node
  -> publishes driving commands during data collection

logger_node
  -> pairs frames with commands
  -> writes images + labels to disk

inference_node
  -> consumes frames
  -> publishes predicted commands

motor_node
  -> converts commands into wheel outputs
```

## Data Collection Mode

In manual mode, a human drives the car while ROS records synchronized camera frames and commands.
This creates a dataset that can later be used for training.

```text
camera_node -> logger_node -> images
teleop_node  -> logger_node -> labels.csv
```

Why this design:

- The camera node stays focused on image capture.
- The teleop node only handles human input.
- The logger owns dataset writing, so the recording logic stays consistent.

## Inference Mode

In inference mode, the human is removed from the loop.
The model predicts commands from live camera frames and the motor node applies them to the chassis.

```text
camera_node -> inference_node -> motor_node -> car
```

Why this design:

- The same ROS structure is reused for both collection and autonomy.
- The inference node can be swapped or retrained without changing the rest of the stack.
- Motor control remains isolated from the learning code.

## Launch Modes

The project uses two launch entrypoints:

```bash
python3 pi/ros/session_launcher.py manual
python3 pi/ros/session_launcher.py inference
```

- `manual` starts data collection
- `inference` starts autonomous driving

## Dataset Output

The logger stores a simple image-and-label dataset:

```text
data/
  images/
    <timestamp>.jpg
  labels.csv
```

Example row:

```text
image,frame_stamp_ns,steering,throttle
1719751234567890123.jpg,1719751234567890123,-0.25,0.60
```

## Model Summary

The model is a small CNN trained from scratch to predict continuous control values.

```text
RGB image -> CNN -> [steering, throttle]
```

The important point is not the exact layer stack, but the behavior:

- it learns directly from driving data
- it outputs continuous control, not discrete labels
- it stays lightweight enough for Raspberry Pi deployment

## Hardware Summary

- Raspberry Pi for onboard execution
- USB camera for front-facing vision
- 2WD chassis kit
- motor driver for wheel control
- separate power supply for motors

## Why This Architecture

- It keeps data collection and autonomy in the same ROS2 framework.
- It separates concerns cleanly: sensing, logging, inference, and actuation.
- It makes future refactors easier because the launch files only decide which nodes are active.

## Full Workflow

```text
1. Human drives the car
2. ROS records frames and labels
3. Dataset is used to train a model
4. Model runs on the Pi
5. Car drives from camera input alone
```

