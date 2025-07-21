# Episode-specific Fine-tuning for Metric-based Few-shot Learners with Optimization-based Training

This repository contains code examples of paper *“Episode‑specific Fine‑tuning for Metric‑based Few‑shot Learners with Optimization‑based Training”* submitted to **IEEE TASLP** by Xuanyu Zhuang, Geoffroy Peeters & Gaël Richard.

🔗 **Paper (arXiv)**:  
(https://www.arxiv.org/abs/2506.17499)

---

## ⚙️ Overview

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

## 🧪 Results

Our approach consistently improved metric-based models (PN, MN, CAN), with ADFT showing particularly strong gains—e.g., a ~5–8 % boost on CAN :contentReference[oaicite:1]{index=1}.

---

## 🚀 Getting started

### 1. Clone

```bash
git clone https://github.com/zdsy/Episode-specific-FT.git
cd Episode-specific-FT
