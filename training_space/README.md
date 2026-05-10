---
title: Cipher Detective Trainer
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: mit
hardware: a10g-small
tags:
  - training
  - cryptography
  - text-classification
  - transformers
---

# 🧠 Cipher Detective — Training Space

This Space auto-trains `roberta-base` on the
[systemslibrarian/classical-cipher-corpus](https://huggingface.co/datasets/systemslibrarian/classical-cipher-corpus)
dataset (81 cipher classes, ~58k balanced training examples) and pushes
the resulting model to
[systemslibrarian/cipher-detective-classifier](https://huggingface.co/systemslibrarian/cipher-detective-classifier).

**Hardware**: A10G small (~24 GB VRAM)  
**Expected time**: 30–60 minutes for 10 epochs  
**Cost**: ~$0.50–1.50 in HF credits

Training starts automatically when the Space boots. The Gradio UI shows
live loss/F1 progress. When done, the Space displays final metrics and
the model is available on the Hub.
