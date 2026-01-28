---
title: "VR Teleoperation for Bimanual Robots"
date: 2024-10-01
draft: false
ShowToc: true
---

This system lets the user control a bimanual robot using a Meta Quest headset. The user puts on the headset, grabs the controllers, and a Trossen AI Mobile robot mirrors their arm movements in real-time through inverse kinematics.

[VIDEO_REAL_ROBOT_PLACEHOLDER: Full teleoperation demo - controlling the real robot]

## Architecture

The system bridges consumer VR hardware to research-grade robotics through a streaming pipeline optimized for low latency.

{{< img src="/img/quest-teleoperation/architecture.svg" alt="System architecture" style="max-width: 450px; display: block; margin: 0 auto;" >}}

The Quest runs a native Android app built with OpenXR that captures controller poses at 60 Hz. Rather than requiring external network infrastructure, the Quest itself runs an embedded MQTT broker. The robot subscribes directly to the headset's IP address, keeping latency under 5ms on a local network.

A key design choice: hand poses are streamed relative to the headset, not in world coordinates. This means if you turn your head, the robot still interprets your gestures from your perspective. You can look around freely without affecting control.

## The Pipeline

{{< video src="/img/quest-teleoperation/questtracking.mp4" autoplay="true" loop="true" muted="true" playsinline="true" style="max-width: 100%; display: block; margin: 0 auto;" >}}

The web interface above shows the Quest streaming pose data in real-time. Control is grip-gated: squeeze to engage, release to disengage. The system captures hand and end-effector poses as references, then applies movement as deltas, so the user can release, reposition, and re-engage without the robot jumping.

[VIDEO_PYROKI_PLACEHOLDER: PyRoki IK visualization - Quest input driving simulated robot]

The IK solver uses PyRoki, a JAX-based differentiable robotics library. Unlike typical 1-step IK solvers, PyRoki optimizes over the entire trajectory, enabling temporal losses that produce smoother, better-aligned motions. A 1-Euro filter further reduces high-frequency tracking noise adaptively.
