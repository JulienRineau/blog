---
title: "GPT-2 Pretraining Implementation"
date: 2023-11-12
draft: false
ShowToc: true
categories: ["recipe"]
---

A nano-GPT implementation with Pytorch Lightning. The goal is to have a clean building block for other research projects by containing just enough manual implementation do be easily modifiable, but also by using common tools to have a painless optimized training and nice monitoring. Its contains the code to train the model, prepare the dataset and run evals. This page also details results I got training on HF's FineWeb-Edu. 

[Code Repository](https://github.com/JulienRineau/gpt2-workflow)


## Architechture

![GPT architechture](/img/gpt-lightning/gpt2-architechture.png)

My implementation is identical to the small GPT2 model but without the dropout layers: 

- 50304 vocab size
- 768 embedding size
- 12 heads 
- 12 transformer block 

This gives us a total of 124M params. 
Its a causal model so next tokens are mask in the self-attention matrix.

```python
GPTLightning(
  (model): GPT(
    (transformer): ModuleDict(
      (wte): Embedding(50304, 768)
      (wpe): Embedding(1024, 768)
      (h): ModuleList(
        (0-11): 12 x Block(
          (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (attn): CausalSelfAttention(
            (c_attn): Linear(in_features=768, out_features=2304, bias=True)
            (c_proj): Linear(in_features=768, out_features=768, bias=True)
          )
          (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (mlp): MLP(
            (c_fc): Linear(in_features=768, out_features=3072, bias=True)
            (gelu): GELU(approximate='tanh')
            (c_proj): Linear(in_features=3072, out_features=768, bias=True)
          )
        )
      )
      (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    )
    (lm_head): Linear(in_features=768, out_features=50304, bias=False)
  )
)
```

## Training
### Optimization Techniques Used

This implementation incorporates several optimization techniques:

- **Scheduled Learning Rate**: Warmup and cosine annealing like in the GPT-3 paper
- **Mixed Precision**: BF16 mixed precision to for computational efficiency and memory usage.
- **Gradient Clipping**: Set to 1.0 like in the GPT-3 paper.
- **Weight Decay**: Set to 0.3, I found that a bigger number than in the GPT-3 paper (0.1) work better in my case.
- **DDP (Distributed Data Parallel)**: Facilitates parallel data processing on multiple GPUs, accelerating training times significantly.

### Dataset
I used a A100 instance from Lambda Labs for the training. The training dataset was a 10B subset of the HF's FineWeb-Edu dataset which is about 3T of disk space. My instance had on 512B of storage the pipeline was built to streamed the dataset, but for better performances storing shard is better.

The dataset loader tokenize FineWeb-edu's docs, then process these tokens into fixed-size chunks based on a predefined sequence length. Each chunk is designed to include an extra token that helps create pairs of inputs and expected outputs. The input consists of a sequence of tokens, and the output is the same sequence shifted by one position.

### Results

Given that I was renting my GPU I did not ran an hyperparemeter sweep but could ran some test on a smaller dataset. I constated that while Mixed Precision and DDP increased the speed, the Scheduled Learning Rate had the biggest impact on the loss. 

![GPT architechture](/img/gpt-lightning/training_loss_chart.png)

The whole training lasted Xh 

Here are some prompting results:

## How to Use This Implementation

For detailed instructions on how to use this implementation, refer to the [GitHub repository](https://github.com/JulienRineau/gpt2-workflow), which includes comprehensive documentation on setup, configuration, and execution.