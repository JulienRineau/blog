---
title: "VR Teleoperation for Bimanual Robots"
date: 2025-01-15
draft: false
ShowToc: true
---

I built a system that lets you control a bimanual robot using a Meta Quest headset. Put on the headset, grab the controllers, and a Trossen AI Mobile robot mirrors your arm movements in real-time through inverse kinematics. The code is available on request.

[VIDEO_REAL_ROBOT_PLACEHOLDER: Full teleoperation demo - controlling the real robot]

## Architecture

The system bridges consumer VR hardware to research-grade robotics through a streaming pipeline optimized for low latency.

<img src="/img/quest-teleoperation/architecture.svg" alt="System architecture" style="max-width: 450px; display: block; margin: 0 auto;">

The Quest runs a native Android app built with OpenXR that captures controller poses at 60 Hz. Rather than requiring external network infrastructure, the Quest itself runs an embedded MQTT broker—the robot subscribes directly to the headset's IP address, keeping latency under 5ms on a local network.

A key design choice: hand poses are streamed relative to the headset, not in world coordinates. This means if you turn your head, the robot still interprets your gestures from your perspective. You can look around freely without affecting control.

## The Pipeline

<video controls style="max-width: 100%; display: block; margin: 0 auto;">
  <source src="/img/quest-teleoperation/questtracking.mp4" type="video/mp4">
</video>

The web interface above shows the Quest streaming pose data in real-time. Control is grip-gated: squeeze to engage, release to disengage. When you squeeze, the system captures your current hand position and the robot's end-effector pose as reference points. Movement is then applied as deltas from these references—you can release, reposition your hands comfortably, and re-engage without the robot jumping.

[VIDEO_PYROKI_PLACEHOLDER: PyRoki IK visualization - Quest input driving simulated robot]

The IK solver uses PyRoki, a JAX-based differentiable robotics library. It runs a least-squares optimization balancing multiple objectives: reaching the target pose, respecting joint limits, staying close to a natural rest configuration, and avoiding shoulder singularities.

Raw tracking data contains high-frequency noise that would cause jittery robot motion. A 1-Euro filter smooths the signal adaptively—heavy filtering during slow movements for stability, reduced filtering during fast movements to preserve responsiveness. The result is smooth motion without perceivable lag.

## What's Next

This teleoperation system serves as data collection infrastructure for imitation learning research. The goal is to record human demonstrations of manipulation tasks, then train neural network policies that can perform these tasks autonomously. The combination of intuitive VR control and a capable bimanual platform makes it practical to collect the diverse demonstration data that learning algorithms need.
