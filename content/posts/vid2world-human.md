---
title: "World Models for Human Manipulation"
date: 2026-01-16
draft: false
ShowToc: true
---

I trained a world model that predicts how humans manipulate objects from a single image and an action sequence.

<video autoplay loop muted playsinline style="width: 100%; max-width: 800px; display: block; margin: 0 auto;">
  <source src="../img/vid2world/comparison.mp4" type="video/mp4">
</video>

*Given the first frame and 16-step action sequence, the model predicts future manipulation frames.*

The premise is simple: if you can accurately simulate what happens when a human performs an action, you don't need a physical robot to learn manipulation. A policy can explore thousands of candidate action sequences in imagination, evaluating outcomes before committing to real-world execution. This isn't just theoretical: 1X Technologies recently demonstrated that world models can drive real humanoid robots [^2], using video diffusion models pretrained on internet-scale data that already understand physics, object permanence, and hand-object interactions. The question becomes: can we steer that knowledge with actions?

## The Core Insight

Standard video diffusion models generate frames that look plausible but ignore what you *want* to happen. They're storytellers, not simulators. To make them useful for robotics, I needed two modifications:

1. **Causality**: Frame 10 shouldn't influence frame 5. The model must respect the arrow of time.
2. **Action conditioning**: The model must understand "if I move my hand here, this happens."

The first sounds trivial—just mask attention—but pretrained video models have bidirectional temporal convolutions baked in everywhere. The second requires the model to learn a new input modality (actions) while preserving its video generation capabilities.

## Architecture

### Training

During training, I have access to full video sequences with paired actions. The key trick: rather than applying uniform noise across all frames, each frame gets an independent noise level. Frame 3 might be nearly clean while frame 12 is heavily corrupted.

Why does this matter? At inference, the model generates autoregressively—it has clean past frames and must generate noisy future frames. By training with variable noise levels, the model learns to leverage clean context to reconstruct corrupted frames. Uniform noise would never teach this skill.

<img src="../img/vid2world/training.svg" alt="Training architecture" style="max-width: 500px; display: block; margin: 0 auto;">

### Inference

At test time, I only have the first frame. Generation proceeds one frame at a time: encode the initial frame, generate frame 2 by denoising conditioned on frame 1, generate frame 3 conditioned on frames 1-2, and so on.

This is where causal attention pays off. Because the model never saw future frames during training, it learned to make predictions from past context alone. KV-caching stores attention keys and values from previously generated frames, so I don't recompute the entire sequence at each step.

<img src="../img/vid2world/inference.svg" alt="Inference architecture" style="max-width: 500px; display: block; margin: 0 auto;">

## Method

The model predicts velocity $v$ rather than noise $\epsilon$—a reparameterization that provides more stable gradients across timesteps [^1]: $v = \sqrt{\bar{\alpha}_t} \cdot \epsilon - \sqrt{1-\bar{\alpha}_t} \cdot x_0$.

The loss sums over frames, each with its own sampled noise level:
\[ \mathcal{L} = \mathbb{E}_{t_1,...,t_{16}} \left[ \sum_{i=1}^{16} \|v_\theta(x_{t_i}, t_i, a_i) - v_{\text{target},i}\|^2 \right] \]

To make actions actually matter, I use classifier-free guidance. During training, I randomly drop actions with 15% probability—but crucially, I drop them *per-frame* rather than per-sequence. This teaches the model fine-grained action-outcome relationships. At inference, I amplify action influence: $v_{\text{guided}} = v_{\text{uncond}} + s \cdot (v_{\text{cond}} - v_{\text{uncond}})$ where $s > 1$.

The trickiest part was converting pretrained weights to causal. Video diffusion models like DynamiCrafter use symmetric temporal convolutions—a kernel that looks at frames before *and after* the current frame. Simply zeroing the future-looking weights destroys learned dynamics. Instead, I used an extrapolative transformation: for a kernel $[w_0, w_1, w_2]$, the causal version becomes $[0, w_0 - w_2, w_1 + 2w_2]$. This preserves the effective temporal receptive field while enforcing strict causality.

## Adapting to Human Manipulation

Most world models target robot arms with 7-DOF action spaces. I wanted to model *human* bimanual manipulation—two hands working together on deformable objects. This required designing a new action representation.

Each hand contributes 10 dimensions: 3D position delta, 6D rotation (two columns of the rotation matrix—more stable than Euler angles or quaternions for learning), and gripper width. The full 20D action captures the coordinated motion of both hands.

| Gripper | Dimensions | Encoding |
|---------|------------|----------|
| Left | 10D | 3D position delta + 6D rotation + 1D width |
| Right | 10D | 3D position delta + 6D rotation + 1D width |

I collected 5,248 episodes of bimanual t-shirt folding using VR teleoperation (details in the [teleoperation post](/posts/quest-teleoperation)). The fisheye camera provides a wide field of view that captures both hands throughout the manipulation. Training ran at 320×512 resolution with batch size 2, gradient accumulation, learning rate 1e-5, and FP16 mixed precision.

## What the Model Learned

The model captures the broad strokes: hand trajectories follow commanded actions, cloth deforms plausibly, spatial relationships stay coherent across frames.

To verify it learned generalizable dynamics rather than memorizing trajectories, I ran an action cross-swap: take frame A with actions from episode B. The grid below shows ground truth (diagonal) versus swapped actions (off-diagonal)—same starting frame, different action sequences producing different outcomes.

<div style="position: relative; max-width: 800px; margin: 0 auto;">
  <div id="carousel" style="display: flex; overflow-x: hidden; scroll-snap-type: x mandatory; scroll-behavior: smooth; border-radius: 8px;">
    <video style="flex: 0 0 100%; scroll-snap-align: start; width: 100%;" autoplay loop muted playsinline>
      <source src="../img/vid2world/crossswap4.mp4" type="video/mp4">
    </video>
    <video style="flex: 0 0 100%; scroll-snap-align: start; width: 100%;" autoplay loop muted playsinline>
      <source src="../img/vid2world/crossswap2.mp4" type="video/mp4">
    </video>
    <video style="flex: 0 0 100%; scroll-snap-align: start; width: 100%;" autoplay loop muted playsinline>
      <source src="../img/vid2world/crossswap3.mp4" type="video/mp4">
    </video>
  </div>
  <button onclick="document.getElementById('carousel').scrollBy({left: -document.getElementById('carousel').offsetWidth})" style="position: absolute; left: 10px; top: 50%; transform: translateY(-50%); background: rgba(0,0,0,0.5); color: white; border: none; border-radius: 50%; width: 40px; height: 40px; font-size: 20px; cursor: pointer;">&#10094;</button>
  <button onclick="document.getElementById('carousel').scrollBy({left: document.getElementById('carousel').offsetWidth})" style="position: absolute; right: 10px; top: 50%; transform: translateY(-50%); background: rgba(0,0,0,0.5); color: white; border: none; border-radius: 50%; width: 40px; height: 40px; font-size: 20px; cursor: pointer;">&#10095;</button>
  <div id="carousel-dots" style="display: flex; justify-content: center; gap: 8px; margin-top: 10px;"></div>
</div>
<script>document.addEventListener("DOMContentLoaded",function(){var c=document.getElementById("carousel"),d=document.getElementById("carousel-dots");if(c&&d){var v=c.querySelectorAll("video");for(var i=0;i<v.length;i++){var dot=document.createElement("button");dot.style.cssText="width:10px;height:10px;border-radius:50%;border:none;background:#666;cursor:pointer";dot.setAttribute("data-i",i);dot.onclick=function(){c.scrollTo({left:c.offsetWidth*this.getAttribute("data-i")})};d.appendChild(dot)}function u(){var idx=Math.round(c.scrollLeft/c.offsetWidth);var dots=d.querySelectorAll("button");for(var j=0;j<dots.length;j++){dots[j].style.background=j===idx?"#fff":"#666"}}c.addEventListener("scroll",u);u()}});</script>

This tracks with findings from 1X: video prediction quality correlates with downstream task success [^2]. If the world model can't accurately predict what happens when you grasp a shirt corner, a policy trained on its rollouts will fail at the real task. Visual fidelity isn't vanity—it's a proxy for physical understanding.

Current limitations reveal what's still hard:
- **Fine details decay**: Fingers blur, cloth texture simplifies over longer horizons. The model takes shortcuts when it can.
- **Complex dynamics**: Cloth folding involves self-collision, layering, contact transitions. The model sometimes produces physically impossible configurations.
- **Depth ambiguity**: Like 1X's monocular system, a single fisheye camera provides weak 3D grounding. The model sometimes confuses depth ordering when hands cross.
- **Large actions**: Big displacements cause hallucination—the model hasn't seen enough extreme motions to generalize.

## What's Next

This world model is infrastructure for the real goal: learning manipulation policies without expensive robot rollouts. The next steps:

- **Inverse dynamics grounding**: Following 1X's architecture, add an inverse dynamics model that extracts action sequences from generated frames. This bridges visual prediction to actionable control.
- **Model-based policy learning**: Train diffusion policies that plan in imagination, using the world model as a simulator.
- **Longer horizons**: Current 16-frame prediction isn't enough for complex tasks. Hierarchical action abstraction or best-of-N sampling at inference could extend temporal reach.

The vision: collect human demonstrations once, train a world model, then train thousands of policies in simulation. Real robot time becomes validation, not training.

---

## References

[^1]: This work builds on video diffusion techniques for world modeling. See [Vid2World: Crafting Video Diffusion Models to Interactive World Models](https://arxiv.org/abs/2505.14357) (Chen et al., 2025). [Project page](https://knightnemo.github.io/vid2world/).

[^2]: 1X Technologies demonstrated world models driving real humanoid robots with minimal robot-specific data. See [World Model for Self-Learning](https://www.1x.tech/discover/world-model-self-learning) (1X, 2025).
