# GPT-CoT

A lightweight fine-tuning project using `phi-2` + LoRA to teach a model how to reason over a grid using Chain-of-Thought (CoT) and simple spatial reasoning.

This project trains a small language model to perform step-by-step 2D navigation using either:
- vector actions like (+1,0)
- NLP directions like "up", "left"

---

## 🧠 Project Goal

Simulate an agent navigating a 10x10 grid using discrete action steps.  
The objective is to compare different input formats and reasoning strategies:
- CoT with vector inputs
- NLP-based commands
- Direct vector-to-position reasoning (baseline)

---

## 🔍 Fine-tuned Models

| Model Folder | Format        | Output                      | Description                                      |
|--------------|---------------|-----------------------------|--------------------------------------------------|
| `phi2-CoT-finetune5`  | `(dx, dy)`    | CoT trace + final pos     | Full reasoning with 5 starting points            |
| `phi2-NLP-finetune1`  | `up/down/...` | CoT trace + final pos     | Instruction-following version                   |
| `phi2-vec-finetune`   | `(dx, dy)`    | Final position only       | Baseline model, no step-by-step explanation     |

Each model is under `outputs/`, and each `.bin` file is under 100MB.

---

## 🛠 Folder Structure

```
GPT-CoT/
├── configs/              # LoRA training config files (YAML)
├── data/                 # JSONL training files
├── outputs/              # Fine-tuned models (3 total)
│   ├── phi2-CoT-finetune5/
│   ├── phi2-NLP-finetune1/
│   └── phi2-vec-finetune/
├── source/               # Training and inference scripts
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🚀 Getting Started

```bash
git clone https://github.com/Seanaaa0/GPT-CoT.git
cd GPT-CoT
conda activate gpt-env  # or your preferred environment
pip install -r requirements.txt
```

---

## 🧪 Inference Example

Input:
```
Actions: (+1,0), (+1,0), (0,+1)
```

Output from `phi2-CoT-finetune5`:
```
Start at (0,0)
Step 1: (0,0) + (+1,0) = (1,0)
Step 2: (1,0) + (+1,0) = (2,0)
Step 3: (2,0) + (0,+1) = (2,1)
Final position: (2,1)
```

---

## 📌 TODO
- [x] Train LoRA on vector trace task
- [x] NLP command version
- [x] Multi-entry point generalization
- [ ] Trace classification (valid/invalid)
- [ ] Decision Transformer for path generation
- [ ] Add goal-aware discriminator

---

## 📜 License
MIT
