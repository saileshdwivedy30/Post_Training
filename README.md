# Supervised Fine-Tuning of Language Models using Hugging Face & TRL

## Project Summary

This repository showcases my implementation and applied understanding of **Supervised Fine-Tuning (SFT)** on causal language models using the Hugging Face Transformers framework alongside the TRL library. The notebook presents both the training workflow and a comparison-based evaluation, organized modularly to highlight key building blocks of the fine-tuning process.

The emphasis of this project is on **gaining conceptual clarity** around the supervised fine-tuning pipeline through structured experimentation, interpretable logging, and working with lightweight models and datasets suited for local development.

---

## 📁 Notebook Structure & Key Components

### 1. Environment Setup
All essential libraries were imported, including `transformers`, `trl`, and `datasets`, along with foundational tools like PyTorch and pandas.

### 2. Helper Functions for Inference and Display
A set of reusable functions was developed to:

- Generate model outputs using prompts structured via `chat_template`.
- Display samples from the dataset with user-assistant roles parsed cleanly.
- Evaluate model performance on a curated list of questions both before and after fine-tuning.

These functions helped isolate logic, improve readability, and enable repeatable testing.

### 3. Baseline Model Evaluation (Before SFT)
A base model (`Qwen3-0.6B-Base`) was evaluated on selected user prompts. This served as a baseline reference for comparison prior to applying fine-tuning.

### 4. Evaluation of Pre-Trained SFT Model
To observe the effect of supervised fine-tuning, I loaded a pre-trained SFT checkpoint (`Qwen3-0.6B-SFT`) and ran it on the same test questions, enabling a clear before-and-after performance comparison.

### 5. Fine-Tuning a Small Model (End-to-End)
Due to compute limitations, I executed full SFT on a smaller model (`SmolLM2-135M`) with a truncated version of a public dataset.

- The dataset was sourced from Hugging Face (`banghua/DL-SFT-Dataset`) and reduced for local execution.
- The tokenizer and model setup followed a consistent structure from earlier steps.

### 6. Training Configuration with `SFTConfig`
I used `trl.SFTConfig` to define all training hyperparameters, including:

- Learning rate and epochs
- Per-device batch size
- Gradient accumulation strategy
- Logging frequency

This section underlines my grasp of how such configurations control training efficiency, stability, and memory requirements.

### 7. Training with `SFTTrainer`
Fine-tuning was conducted using `SFTTrainer`, integrating the model, config, dataset, and tokenizer into a minimal loop. The training process was intentionally lightweight to ensure full interpretability and local reproducibility.

### 8. Evaluation Post-Fine-Tuning
Once training completed, the newly fine-tuned model was evaluated on the original question set through manual response comparison. The observed output shifts offered qualitative insight into how SFT altered the model’s behavior.

---

## Interpretable Notebook:

Rather than focusing on scale, this project prioritizes **clarity and reproducibility**. The training process is kept intentionally small so that it can be:

- Understood through `print()`-level inspection and visual checks
- Interacted with step by step to demystify fine-tuning mechanics
- Easily adapted to larger setups once the fundamentals are clear

The result is a project designed to emphasize **technical correctness and hands-on comprehension**, entirely executable on standard local machines.

---
