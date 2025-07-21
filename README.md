# Episode-specific Fine-tuning for Metric-based Few-shot Learners with Optimization-based Training

This repository contains code examples of paper *“Episode‑specific Fine‑tuning for Metric‑based Few‑shot Learners with Optimization‑based Training”* submitted to **IEEE TASLP** by Xuanyu Zhuang, Geoffroy Peeters & Gaël Richard.

🔗 **Paper (arXiv)**:  
(https://www.arxiv.org/abs/2506.17499)

---

## Overview

- Implements three *episode-specific fine-tuning* methods:
  - **RDFT** (Rotational Division Fine-Tuning)
  - **IDFT** (Iterative Division Fine-Tuning)
  - **ADFT** (Augmented Division Fine-Tuning)
- Integrates a meta-learning training paradigm based on **Meta‑Curvature**.
- Tested on three audio datasets:
  - **ESC‑50** (environmental sounds)
  - **Speech Commands V2** (spoken keywords)
  - **Medley‑solos‑DB** (musical instruments)

---

## Repo Structure

The repo contains implementations of 3 different model-dataset combinations, with each of the combination includes the training and testing of all three proposed episode-specific fine-tuning methods (RDFT, EDFT & ADFT). Among which, ADFT methods (integrating audio augmentations) uses a separate training pipeline to handle wavform loading and switching between augmentation methods.

## Setup
```bash
git clone https://github.com/zdsy/Episode-specific-FT.git
cd Episode-specific-FT
pip install -r requirements.txt
```

To run the examples, simply change to the target example directory and run:
```bash
python train.py
```
---

## Cite
```bash
@misc{zhuang2025episodespecificfinetuningmetricbasedfewshot,
      title={Episode-specific Fine-tuning for Metric-based Few-shot Learners with Optimization-based Training}, 
      author={Xuanyu Zhuang and Geoffroy Peeters and Gaël Richard},
      year={2025},
      eprint={2506.17499},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.17499}, 
}
```

---

## Acknowledgments

The implementation uses the [**learn2learn**](https://github.com/learnables/learn2learn) library for few-shot meta-learning.


