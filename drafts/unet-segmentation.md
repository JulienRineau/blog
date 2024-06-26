---
title: "U-Net Segmentation"
date: 2023-09-13
draft: false
ShowToc: true
---
A simple Pytroch U-net implementation. The goal is to have an clean building block that can be used in other bigger projects (e.g. Diffusion). The net is tested with a segmentation task on the MIT scene-parse-150 dataset.

[Code Repository](https://github.com/JulienRineau/unet-segmentation)

## Architechture

![GPT architechture](/img/unet-segmentation/transformer-unet-architecture.png)

My implementation is based on the graph above:
- 8 double-conv blocks with batch-norm and GELU for activation 
- 4 skip connections

