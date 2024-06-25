---
title: "Paper Presentations Generator"
date: 2023-07-30
draft: false
ShowToc: true
---
A summer project in which I built an app that generate video presentations with voice of a scientific paper because keeping up with daily papers take too much time. I plan to automatically publish it on X so everyone can enjoy it but in the meantime ppl can just fork the repo and replicate run it locally.

[Code Repository](https://github.com/JulienRineau/unet-segmentation)

## The Stack
The backend stack used is:
- **GPT4-V** for paper understanding
- **Unstructured.io** for images extractions
- **Whisper** for speech
- **Reveal.js** for presentations
- **Movie.py** for the video and text superposition
- **Flask** for serving

The front end stack is just **Next.js** with **Tailwind.css**.