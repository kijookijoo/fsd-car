# 🚗 End-to-End Self-Driving RC Car (From-Scratch CNN)

A real-time autonomous RC car built using a **custom convolutional neural network trained from scratch**, using self-collected driving data.

The system learns to map raw camera images directly to control signals (steering + throttle) using behavioral cloning.


---

# 🧠 Core Idea

Instead of hand-engineered rules or pretrained models, the system learns driving behavior directly from data:

```
Image → CNN (from scratch) → Steering + Throttle
```

The model learns by imitating human driving.

---

# 🏗️ System Architecture

## 1. Data Collection (Teleoperation Mode)

You manually drive the RC car while the system records data.

```
Raspberry Pi (SSH session)
        ↓
Camera captures frames
        ↓
Human drives car (keyboard / controller)
        ↓
System logs:
    image + steering + throttle
        ↓
Dataset stored locally
```

### Dataset structure

```
dataset/
 ├── images/
 │    000001.jpg
 │    000002.jpg
 └── labels.csv
```

### CSV format

```
image,steering,throttle
000001.jpg,-0.25,0.60
000002.jpg, 0.10,0.55
```

---

## 2. Training Pipeline (Laptop)

A custom CNN is trained from scratch (no pretrained weights).

```
Image → Custom CNN → [steering, throttle]
```

Example output:

```
tensor([-0.32, 0.58])
```

---

## 3. Deployment Pipeline (Raspberry Pi)

```
Live Camera Feed
        ↓
Image Preprocessing (resize, normalize)
        ↓
Custom CNN Inference
        ↓
Control Output (steering + throttle)
        ↓
Motor Driver (TB6612FNG)
        ↓
DC Motors → RC Car movement
```

---

# 🧩 Hardware Components

## 🧠 Compute
- Raspberry Pi 5 (4GB RAM recommended)
- microSD Card (32GB–128GB)
- USB-C Power Supply (5V / 5A recommended)

## 📷 Vision
- Logitech USB Webcam (720p or 1080p)
- Optional camera mount

## 🚗 Chassis & Motion
- 2WD RC Robot Car Chassis Kit
- 2 DC motors + wheels
- Caster wheel

## ⚡ Motor Control
- TB6612FNG motor driver
- Jumper wires + breadboard

## 🔋 Power System
- Battery pack for motors (separate from Pi)
- Raspberry Pi powered independently

## 🎮 Data Collection Input
Used ONLY during training:
- USB Game Controller (recommended)
- or Keyboard (WASD)

## ❄️ Cooling
- Raspberry Pi 5 active cooler (recommended)
- Passive heatsink (minimum acceptable)

---

# 🧠 Model Architecture (From Scratch CNN)

## Input
```
RGB Image (224 × 224 × 3)
```

## Output
```
[steering, throttle]
```

Example:
```
tensor([-0.30, 0.55])
```

---

## Architecture

```
Conv2D (3 → 16) + ReLU
Conv2D (16 → 32) + ReLU
Conv2D (32 → 64) + ReLU
Flatten
Linear → 128
Linear → 2 outputs
```

---

## Loss Function
```
Smooth L1 Loss (Huber Loss)
```

## Optimizer
```
Adam (lr=1e-4)
```

---

# 📊 Dataset Format

```
image, steering, throttle
000001.jpg, -0.20, 0.60
000002.jpg,  0.10, 0.55
000003.jpg,  0.00, 0.40
```

---

# 🔄 Full Workflow

## Step 1 — Data Collection

```
Human drives car
        ↓
Camera captures frames
        ↓
Controls recorded (steering + throttle)
        ↓
Dataset created
```

---

## Step 2 — Training

```
Dataset → CNN (from scratch) → Driving policy learned
```

---

## Step 3 — Deployment

```
Camera → CNN → Motor commands → Car drives autonomously
```

---

# ⚠️ Key Design Principles

## ✔ No pretrained models
All features learned from scratch.

## ✔ End-to-end learning
No intermediate modules:

- No lane detection  
- No object detection  
- No rule-based control  

## ✔ Direct control prediction

Instead of:
```
LEFT / RIGHT / STOP
```

we use:
```
continuous steering + throttle
```

---

# 📌 Known Limitations

- Requires consistent dataset quality
- Sensitive to lighting changes
- Performance depends on driving behavior during data collection
- Can fail outside training environment

---

# 🚀 Future Improvements

- Temporal model (CNN + LSTM)
- Obstacle detection (YOLO-based)
- PID smoothing controller
- Multi-camera setup
- Reinforcement learning fine-tuning

---

# 🧠 Learning Goals

This project demonstrates:

- PyTorch from-scratch CNN design
- Behavioral cloning
- Dataset engineering
- Real-time inference on edge devices
- Embedded ML deployment
- Robotics integration

---

# 🚗 Final System Summary

```
YOU DRIVE → DATASET → CNN LEARNS → CAR IMITATES YOU
```