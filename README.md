# GPT-CoT

A lightweight fine-tuning project using `phi-2` + LoRA to teach a model how to reason over a grid using Chain-of-Thought (CoT).

This project trains a small language model to perform step-by-step 2D vector addition, such as navigating a 10x10 grid using directional vectors (e.g., (+1,0), (0,+1)).

## 🔧 Features
- Chain-of-Thought format training using Alpaca-style data
- Inference script with step-by-step trace output
- Handles fallback when Final position is missing
- Plans for NLP-format directions and Decision Transformer support

## 🗂 Folder Structure

```
GPT-CoT/
├── configs/              # LoRA training config files (YAML)
├── data/                 # CoT-style JSONL training data
├── source/               # Python scripts (train, inference, generation)
├── .gitignore
├── README.md
└── requirements.txt
```

## 🚀 Getting Started

```bash
git clone https://github.com/Seanaaa0/GPT-CoT.git
cd GPT-CoT
conda activate gpt-env  # or your preferred environment
pip install -r requirements.txt
```

## 🧠 Example Task

Input:
```
Actions: (+1,0), (+1,0), (0,+1)
```

Output:
```
Start at (0,0)
Step 1: (0,0) + (+1,0) = (1,0)
Step 2: (1,0) + (+1,0) = (2,0)
Step 3: (2,0) + (0,+1) = (2,1)
Final position: (2,1)
```

## 📌 TODO
- [x] LoRA training on phi-2 with vector trace task
- [ ] Add (9,9) starting point (up/left direction)
- [ ] Convert vector actions to NLP ("left", "right", ...)
- [ ] Trace classification (valid / invalid)
- [ ] Decision Transformer path generation

## 📜 License
MIT
