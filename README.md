
# GPT-CoT

A lightweight fine-tuning project using `phi-2` + LoRA to teach a model how to reason over a grid using Chain-of-Thought (CoT) and simple spatial reasoning.

This project trains a small language model to perform step-by-step 2D navigation using either:
- vector actions like (+1,0)
- NLP directions like "up", "left"

---

##  Project Goal

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


###  New Models (August 2025)

| Model Folder                  | Format        | Output                      | Description                                                  |
|------------------------------|---------------|-----------------------------|--------------------------------------------------------------|
| `phi2-CoT-finetune11x11`     | `(dx, dy)`    | CoT trace + final pos       | Trained on 11x11 map-free world, perfect accuracy            |
| `phi2-CoT-finetune11x11_map` | `(dx, dy)`    | CoT trace + final pos + SG map | Input includes grid map with S, model returns final map with SG |
| `phi2-Label-finetune1`            | `(dx, dy)`    | CoT trace + label           | Labeled path validity on map with wall (future extension)    |

###  Dataset Labels

- `11x11`: basic spatial trace task, vector action → position (no map)
- `11x11 map`: adds map context to input, model must parse visual structure
- `label`: data includes correctness classification (`correct`, `loop`, etc.)

## 🛠 Folder Structure

```
GPT-CoT/
├── configs/              # LoRA training config files (YAML)
├── data/                 # JSONL training files
├── outputs/              # Fine-tuned models (3 total)
│   ├── phi2-CoT-finetune5/
│   ├── phi2-CoT-finetune11x11/
│   ├── phi2-CoT-finetune11x11_map/
│   ├── phi2-Label-finetune1/
│   ├── phi2-NLP-finetune1/
│   └── phi2-vec-finetune/
├── source/               # Training and inference scripts
├── .gitignore
├── README.md
└── requirements.txt
```

---

##  Getting Started

```bash
git clone https://github.com/Seanaaa0/GPT-CoT.git
cd GPT-CoT
conda activate gpt-env  # or your preferred environment
pip install -r requirements.txt
```

---

##  August 2025 Updates

###  Label-based Reasoning Model

-  New fine-tuned model: `phi2-Label-finetune1`
-  Task: Given a series of vector actions `(dx,dy)`, reason step-by-step to compute the final position and classify the path as one of:
- `correct`, `too short`, `too long`, `loop`, `out of bound`, `wrong`
-  Training data: `10x10_vec_labeled.jsonl`
-  Inference script: `inference_phi2_vec.py`
-  Accuracy: ~95%, supports full CoT + label correctness tracking
-  Output example includes `"label"` and `"correct"` field for each prediction

###  Interactive Web Interface

-  `map_interface.html`: displays a 10x10 grid and agent paths interactively
-  `flask_api.py`: serves model predictions and links frontend ↔ backend
-  Future integration with live inference and editing

##  Trace Visualization Tool

We provide a Python tool to visualize inference traces from test_label.jsonl.

###  Script: `generate_trace_images.py`

This script will:
- Parse GPT output traces
- Generate per-sample visualizations
- Combine up to 25 images into a grid

##  Trainer Metric Visualization

Use `plot_trainer_state_cli.py` to visualize training loss and gradients:

```bash
python plot_trainer_state_cli.py --file results/trainer_state/trainer_stateX.json --metrics 1
```

- Metric options:
  - `1`: Loss
  - `2`: Grad norm
  - `3`: Learning rate

Output PNG files are saved to `results/png/`.

---


###  Usage
```bash
cd source/data/test_output
python generate_trace_images.py

---

##  TODO
- [✅] Train LoRA on vector trace task
- [✅] NLP command version
- [✅] Multi-entry point generalization
- [✅] Trace classification (valid/invalid)
- [ ] Decision Transformer for path generation
- [ ] Add goal-aware discriminator

---

## 📜 License
MIT
