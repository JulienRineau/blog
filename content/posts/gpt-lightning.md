---
title: "GPT-Lightning"
date: 2024-06-12
draft: false
ShowToc: true
---

A nano-GPT implementation with Pytorch Lightning. The goal is to have a clean building block for other research projects by containing just enough manual implementation do be easily modifiable, but also by using common tools to have a painless optimized training and nice monitoring. Its contains the code to train the model, prepare the dataset and run evals. This page also details results I got training on HF's FineWeb-Edu. 

[Code Repository](https://github.com/JulienRineau/gpt2-workflow)

![GPT architechture](/img/gpt-lightning/gpt2-architechture.png)

## Optimization Techniques Used

This implementation incorporates several advanced optimization techniques to improve training efficiency and model performance:

- **Scheduled Learning Rate**: Implements warmup and cosine annealing to adjust the learning rate throughout training.
- **Mixed Precision**: Utilizes BF16 mixed precision to enhance computational efficiency and reduce memory usage.
- **Gradient Clipping**: Helps in avoiding exploding gradients problem by clipping the gradients to 1.0.
- **Weight Decay**: Adds a regularization term to the loss to prevent overfitting.
- **DDP (Distributed Data Parallel)**: Facilitates parallel data processing on multiple GPUs, accelerating training times significantly.

## Training results
![GPT architechture](/img/gpt-lightning/training_loss_chart.png)

## How to Use This Implementation

For detailed instructions on how to use this implementation, refer to the [GitHub repository](https://github.com/JulienRineau/gpt2-workflow), which includes comprehensive documentation on setup, configuration, and execution.