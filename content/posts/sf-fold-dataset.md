---
title: "Large-Scale Robotics Data Collection with UMI-Style Grippers"
date: 2025-11-01
draft: false
ShowToc: true
---

<p style="text-align: center; color: #888; font-size: 0.85rem; margin-bottom: 1.5rem;">(Additional contributors, alphabetically: Laz Kopits, Ryan Leong, Sheel Taskar, Daryl Yang, William Zang)</p>

Building robust robotic manipulation policies requires diverse, high-quality demonstration data at scale. The challenge lies not just in collecting data, but in doing so reliably across multiple devices while maintaining temporal synchronization and handling the inevitable network disruptions of real-world deployments. This post details the system architecture behind the **[SF Fold dataset](https://huggingface.co/datasets/zeroshotdata/sf_fold)**, a large-scale robotics manipulation dataset collected using UMI-style grippers.

<video autoplay loop muted playsinline style="max-width: 100%; display: block; margin: 0 auto;">
  <source src="../img/sf-fold/demo_video_web.mp4" type="video/mp4">
</video>

> **Our Goal**: Take the UMI concept from research prototype to **production-ready hardware**: a device anyone can pick up and use without specialized expertise, seamlessly integrated into cloud-based post-processing pipelines for scalable data collection.

> **Dataset**: [huggingface.co/datasets/zeroshotdata/sf_fold](https://huggingface.co/datasets/zeroshotdata/sf_fold)

---

## Hardware Setup

The data collection system builds on the [UMI (Universal Manipulation Interface)](https://umi-gripper.github.io/) approach, using UMI-style still grippers equipped with OAK-D Wide cameras for 6-DoF pose tracking via ORB-SLAM3 in stereo mode. The setup supports bimanual manipulation tasks through a **multi-puppet configuration**: Left Puppet, Right Puppet, and Ego Puppet. Throughout this document, we refer to the data collection devices as "puppets".

| Component | Specification |
|-----------|--------------|
| Gripper | UMI-Style Still Gripper |
| Camera | OAK-D Wide |
| IMU/Hall Sensors | 200Hz sampling rate |
| Camera Frame Rate | 30Hz |
| Control Interface | Physical button + LED feedback |

> **Button Controls**: *Long press* creates/closes datasets, *short press* starts recording episodes. Color-coded LEDs display current system status.

### OAK-D Wide Camera

The OAK-D Wide provides stereo vision for 6-DoF pose tracking using ORB-SLAM3. The camera is mounted pointing upward on each gripper to avoid occlusions during manipulation tasks.

<div id="oak-d-viewer" style="width: 100%; height: 400px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 8px; margin: 1rem 0; position: relative; cursor: grab;">
  <div id="viewer-loading" style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #888; font-family: system-ui;">Loading 3D model...</div>
</div>
<p style="text-align: center; font-style: italic; color: #666; margin-top: -0.5rem;">Interactive 3D model of the OAK-D Wide camera. Click and drag to rotate, scroll to zoom.</p>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/OBJLoader.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
(function() {
  const container = document.getElementById('oak-d-viewer');
  const loading = document.getElementById('viewer-loading');

  // Scene setup
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.001, 1000);
  camera.position.set(0.25, 0.15, 0.25);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(0x000000, 0);
  container.appendChild(renderer.domElement);

  // Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(ambientLight);

  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
  directionalLight.position.set(5, 10, 7);
  scene.add(directionalLight);

  const backLight = new THREE.DirectionalLight(0x4a90d9, 0.4);
  backLight.position.set(-5, -5, -5);
  scene.add(backLight);

  // Controls
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.rotateSpeed = 0.8;
  controls.enableZoom = true;
  controls.minDistance = 0.05;
  controls.maxDistance = 0.8;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 1.0;

  // Load OBJ
  const loader = new THREE.OBJLoader();
  loader.load('../img/sf-fold/oak_d_wide.obj',
    function(object) {
      // Center the model
      const box = new THREE.Box3().setFromObject(object);
      const center = box.getCenter(new THREE.Vector3());
      object.position.sub(center);

      // Apply material
      object.traverse(function(child) {
        if (child instanceof THREE.Mesh) {
          child.material = new THREE.MeshPhongMaterial({
            color: 0xaaaaaa,
            specular: 0x555555,
            shininess: 40
          });
        }
      });

      scene.add(object);
      loading.style.display = 'none';
    },
    function(xhr) {
      const percent = Math.round((xhr.loaded / xhr.total) * 100);
      loading.textContent = 'Loading 3D model... ' + percent + '%';
    },
    function(error) {
      loading.textContent = 'Error loading model';
      console.error('OBJ load error:', error);
    }
  );

  // Animation loop
  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  // Handle resize
  window.addEventListener('resize', function() {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  });

  // Stop auto-rotate on interaction
  container.addEventListener('mousedown', () => controls.autoRotate = false);
  container.addEventListener('touchstart', () => controls.autoRotate = false);
})();
</script>

---

## System Architecture

The system follows a distributed architecture where multiple puppets coordinate through a central MQTT broker running on the Ego Puppet. This enables synchronized multi-view data collection while maintaining robustness against network failures.

<img src="../img/sf-fold/puppet_communication.png" alt="Puppet Communication Architecture" class="zoomable-img" style="width: 100%; max-width: 900px; display: block; margin: 1.5rem auto; border-radius: 4px; cursor: zoom-in;">
<p style="text-align: center; font-style: italic; color: #666; margin-top: -0.5rem;">Architecture showing Right, Ego, and Left Puppets communicating via the central EgoBroker. Components include MQTTCommandListener, ButtonNode, and sensor_app. <em>(Click to enlarge)</em></p>

### Puppet Communication

Puppet interaction follows a state machine: `ON → DATASET_ACTIVE → RECORDING → UPLOADING`

Each puppet runs three core components:
- **MQTTCommandListener**: Receives start/stop commands, triggers local recording scripts
- **ButtonNode**: Handles physical button input on Right Puppet, broadcasts commands
- **sensor_app**: Publishes sensor data and system status

When a user presses the button on the Right Puppet, the ButtonNode publishes to the Ego Broker, which distributes synchronized commands to all puppets, ensuring simultaneous state transitions and episode alignment.

### MQTT Broker Setup

The system implements a **dual-broker architecture** balancing reliability with discovery:

| Broker | Port | Purpose |
|--------|------|---------|
| Cloud (GCS) | 1883 | Discovery only |
| Local (Mosquitto) | 1884 | All internal communications |

> **Key Design**: The cloud broker serves *only* as a discovery mechanism. All data recording and episode management happens through the local broker, allowing fully offline operation after initial setup.

<details>
<summary><strong>Cloud Broker Details</strong></summary>

- Runs on Google Cloud infrastructure
- Each puppet publishes system metrics to `<GROUP_NAME>/<PUPPET_SIDE>/system_metrics`
- Metrics include broker status and IP address
- New puppets subscribe to discover their local broker IP
- Connection uses standard port 1883 with admin credentials
- Automatic fallback to searching mode if connection fails

</details>

<details>
<summary><strong>Local Broker Details</strong></summary>

- Mosquitto MQTT broker instance
- Handles all internal communications:
  - Message routing between system components
  - Episode transitions and sensor data
  - State change notifications
- Maintains message order and delivery during disconnections
- Functions independently of internet connectivity
- All system events pass through this broker

</details>

---

## Data Collection Flow

The data flow spans from physical demonstration through cloud storage, with multiple checkpoints for reliability.

<img src="../img/sf-fold/sequence_diagram.png" alt="Data Flow Sequence Diagram" class="zoomable-img" style="width: 100%; max-width: 900px; display: block; margin: 1.5rem auto; border-radius: 4px; cursor: zoom-in;">
<p style="text-align: center; font-style: italic; color: #666; margin-top: -0.5rem;">Complete data flow between Demonstrator, Puppet, MQTT Brokers, Flask API, Cloud SQL, GCS, and Pub/Sub. <em>(Click to enlarge)</em></p>

### Status Publishing

Each puppet continuously publishes status (IP, health metrics) to the Cloud MQTT Broker, enabling system monitoring and discovery.

### Acquisition Flow

When the demonstrator presses the start/stop button:
1. Button press publishes command to local MQTT broker
2. Command triggers acquisition scripts on all connected puppets
3. Each puppet records sensor data with precise timestamps

### Upload Mechanism

After recording completes, uploads follow a secure, verifiable sequence:

| Step | Endpoint | Action |
|------|----------|--------|
| 1 | `POST /puppet/request_upload` | Request upload, log to Cloud SQL |
| 2 | - | Flask API generates signed URL |
| 3 | `PUT` to GCS | Upload recording with signed URL |
| 4 | `POST /puppet/confirm_upload` | Confirm completion in Cloud SQL |
| 5 | Pub/Sub | Trigger post-processing pipeline |

<details>
<summary><strong>Upload Flow Details</strong></summary>

1. **Request Upload**: Puppet initiates via `POST /puppet/request_upload` to Flask API, which logs "Upload Requested" in Cloud SQL

2. **Signed URL Generation**: Flask API generates a signed URL with time-limited write access to the GCS bucket

3. **Data Upload**: Puppet uploads directly to GCS using `PUT` with the signed URL

4. **Confirm Upload**: Puppet confirms via `POST /puppet/confirm_upload`, updating Cloud SQL to "Upload Confirmed"

5. **Trigger Processing**: Completion publishes to Pub/Sub topics (`RawDataForCalibration` or `RawData`) for downstream pipelines

This mechanism ensures data integrity through explicit confirmation steps and enables tracking through Cloud SQL status entries.

</details>

---

## Reliability and Synchronization

Collecting synchronized multi-view data across distributed devices requires robust handling of network failures.

### Reconnection Mechanism

TCP connections can become stale without detection, causing silent disconnections. The system employs two mechanisms:

| Mechanism | Function |
|-----------|----------|
| **TCP Keepalive** | Periodic health checks, detects broken connections even when idle |
| **Socket Timeouts** | 5-second max wait, prevents indefinite hangs |

> **Failure Behavior**: When connections fail, puppets continue recording locally while tracking disconnection events. LED shows `CONNECTING_TO_BROKER` (pulsing white) during recovery.

<details>
<summary><strong>TCP Keepalive Details</strong></summary>

- Activates periodic connection health checks
- Automatically sends small test messages when connection is idle
- Detects broken connections even when no data is being sent
- Configured when each puppet first connects to the broker

</details>

<details>
<summary><strong>Socket Timeout Details</strong></summary>

- Sets maximum waiting time of 5 seconds for all network operations
- If sending/receiving data exceeds threshold, connection marked as failed
- Prevents system from hanging indefinitely during communication attempts
- Configured at initial connection establishment

</details>

### Episode Synchronization

When a puppet disconnects during recording, it misses episode transitions, causing numbering to become unsynchronized. The reconnection mechanism operates in **three phases**:

| Phase | Action |
|-------|--------|
| **1. Disconnect** | Store episode number in `/tmp/episode_at_disconnect`, log to `episode_timestamps.txt` |
| **2. Reconnection** | Obtain current system-wide episode from broker, compare with saved |
| **3. Synchronization** | If mismatch: broadcast `sync_episode` command, align all counters |

<details>
<summary><strong>Phase Details</strong></summary>

**1. Disconnect Phase**
- Current episode number stored in `/tmp/episode_at_disconnect`
- Disconnection timestamp recorded in `episode_timestamps.txt`
- Format: `# DISCONNECTED FROM BROKER at: [timestamp]`
- Recording continues in current episode despite disconnection

**2. Reconnection Phase**
- Current system-wide episode number obtained from broker
- Reconnection timestamp recorded in `episode_timestamps.txt`
- Format: `# RECONNECTED TO BROKER at: [timestamp]`
- Saved episode number compared with current system episode

**3. Synchronization Phase**
- If episode numbers match: continue recording in current episode
- If mismatch detected:
  - System broadcasts a `sync_episode` command
  - All puppets process this as standard episode transition
  - New episode timestamp added to `episode_timestamps.txt`
  - Episode counters aligned across all puppets

</details>

<details>
<summary><strong>Example: Synchronized Timestamps</strong></summary>

When properly synchronized, all puppets maintain identical episode logs:

```
ego/episode_timestamps.txt:
Episode 1 started at: 1740348759.123456789
Episode 2 started at: 1740348852.987654321
Episode 3 started at: 1740348912.456789123
Episode 4 started at: 1740348975.321654987
Episode 5 started at: 1740349032.789456123
Episode 6 started at: 1740349102.654789321
```

</details>

<details>
<summary><strong>Example: Disconnection Scenarios</strong></summary>

The `left/episode_timestamps.txt` below shows two disconnection events:

```
Episode 1 started at: 1740348759.123456789
# DISCONNECTED FROM BROKER at: 1740348800.123456789
# RECONNECTED TO BROKER at: 1740348920.654321987
Episode 3 started at: 1740348925.789123456
Episode 4 started at: 1740348975.321654987
# DISCONNECTED FROM BROKER at: 1740349000.123456789
# RECONNECTED TO BROKER at: 1740349040.654321987
Episode 5 started at: 1740349045.789123456
Episode 6 started at: 1740349102.654789321
```

---

**First Disconnection (Skipping Episodes)**:
1. All puppets start Episode 1 together
2. Left puppet disconnects at timestamp `1740348800`
3. While disconnected, ego and right transition to Episodes 2 and 3
4. Left reconnects at `1740348920`, others are in Episode 3
5. Left syncs to Episode 3, *completely skipping Episode 2*
6. Timestamp shows Episode 3 starting shortly after reconnection (`1740348925`)

---

**Second Disconnection (Single Episode Skip)**:
1. All puppets record Episode 4 together
2. Left disconnects at `1740349000`
3. While disconnected, ego and right transition to Episode 5
4. Left reconnects at `1740349040`, others are in Episode 5
5. Left syncs to Episode 5 at `1740349045`
6. All puppets transition to Episode 6 together

</details>

### Timing and Clock Synchronization

Precise timing is crucial for multi-device synchronization. When a new episode starts, the system records the exact timestamp to `episode_timestamps.txt`. Episode boundaries are determined by comparing each data point's timestamp with recorded start times.

| Sensor Type | Sample Rate | Transition Gap | Missing Samples |
|-------------|-------------|----------------|-----------------|
| IMU/Hall | 200Hz | ~10μs | 1-2 samples |
| Camera | 30Hz | ~20ms | 0-1 frames |

> **Clock Sync**: All devices use **Chrony** with public NTP servers, providing sub-millisecond accuracy for multi-device coordination.

<details>
<summary><strong>Timing Details</strong></summary>

**Episode Boundary Detection**:
- Data points with timestamps *before* episode start → previous episode
- Data points with timestamps *after or equal* → new episode

**IMU/Hall Sensors (200Hz)**:
- Typical gap around episode transition: ~10 microseconds
- Usually missing 1-2 samples at transition
- Last sample typically ~5μs before transition
- First new sample typically ~5μs after transition

**Camera Data (30Hz)**:
- Typical gap around episode transition: ~20 milliseconds
- Usually missing 0-1 frames at transition
- Last frame typically ~10ms before transition
- First new frame typically ~10ms after transition

**Clock Synchronization**:
- Chrony installed on each Raspberry Pi host
- Continuously synchronizes with reliable external NTP sources
- Provides sub-millisecond accuracy sufficient for multi-device coordination

</details>

---

## Sample Data Visualization

The dataset includes interactive 3D visualizations using [Rerun](https://rerun.io/), allowing exploration of recorded episodes with synchronized multi-view camera feeds, gripper poses, and sensor data.

<details>
<summary><strong>View Sample Episodes in Rerun (10 samples)</strong></summary>

| Sample | Link |
|--------|------|
| 1 | [Rerun Viewer #1](https://app.rerun.io/version/0.24.1/?url=https://storage.googleapis.com/zeroshot-public-rrds/puppet-high-res-rrd/1.rrd&renderer=webgl) |
| 2 | [Rerun Viewer #2](https://app.rerun.io/version/0.24.1/?url=https://storage.googleapis.com/zeroshot-public-rrds/puppet-high-res-rrd/2.rrd&renderer=webgl) |
| 3 | [Rerun Viewer #3](https://app.rerun.io/version/0.24.1/?url=https://storage.googleapis.com/zeroshot-public-rrds/puppet-high-res-rrd/3.rrd&renderer=webgl) |
| 4 | [Rerun Viewer #4](https://app.rerun.io/version/0.24.1/?url=https://storage.googleapis.com/zeroshot-public-rrds/puppet-high-res-rrd/4.rrd&renderer=webgl) |
| 5 | [Rerun Viewer #5](https://app.rerun.io/version/0.24.1/?url=https://storage.googleapis.com/zeroshot-public-rrds/puppet-high-res-rrd/5.rrd&renderer=webgl) |
| 6 | [Rerun Viewer #6](https://app.rerun.io/version/0.24.1/?url=https://storage.googleapis.com/zeroshot-public-rrds/puppet-high-res-rrd/6.rrd&renderer=webgl) |
| 7 | [Rerun Viewer #7](https://app.rerun.io/version/0.24.1/?url=https://storage.googleapis.com/zeroshot-public-rrds/puppet-high-res-rrd/7.rrd&renderer=webgl) |
| 8 | [Rerun Viewer #8](https://app.rerun.io/version/0.24.1/?url=https://storage.googleapis.com/zeroshot-public-rrds/puppet-high-res-rrd/8.rrd&renderer=webgl) |
| 9 | [Rerun Viewer #9](https://app.rerun.io/version/0.24.1/?url=https://storage.googleapis.com/zeroshot-public-rrds/puppet-high-res-rrd/9.rrd&renderer=webgl) |
| 10 | [Rerun Viewer #10](https://app.rerun.io/version/0.24.1/?url=https://storage.googleapis.com/zeroshot-public-rrds/puppet-high-res-rrd/10.rrd&renderer=webgl) |

</details>

---

## Dataset and Usage

The SF Fold dataset is available on Hugging Face:

> **Download**: [huggingface.co/datasets/zeroshotdata/sf_fold](https://huggingface.co/datasets/zeroshotdata/sf_fold)

The dataset focuses on folding and manipulation tasks, providing synchronized multi-view recordings suitable for:

| Application | Description |
|-------------|-------------|
| **Imitation Learning** | Training policies from human demonstrations |
| **Diffusion Policy** | Learning manipulation behaviors through diffusion models |
| **Multi-view Fusion** | Algorithms leveraging multiple camera perspectives |
| **Temporal Modeling** | Models reasoning about action sequences over time |

---

*For more details on the original UMI approach that inspired this work, see the [paper](https://arxiv.org/abs/2402.10329) and [project page](https://umi-gripper.github.io/).*

<style>
.img-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.9);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 9999;
  cursor: zoom-out;
}
.img-overlay img {
  max-width: 95%;
  max-height: 95%;
  object-fit: contain;
}
</style>

<script>
document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('.zoomable-img').forEach(function(img) {
    img.addEventListener('click', function() {
      const overlay = document.createElement('div');
      overlay.className = 'img-overlay';
      const enlargedImg = document.createElement('img');
      enlargedImg.src = img.src;
      enlargedImg.alt = img.alt;
      overlay.appendChild(enlargedImg);
      document.body.appendChild(overlay);
      document.body.style.overflow = 'hidden';
      overlay.addEventListener('click', function() {
        overlay.remove();
        document.body.style.overflow = '';
      });
    });
  });
});
</script>
