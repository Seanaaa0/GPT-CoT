---
library_name: peft
tags:
- generated_from_trainer
datasets:
- /home/seana/axolotl_project/data/simple_10x10_1to2000.jsonl
base_model: /home/seana/axolotl_project/models/phi-2/phi2
model-index:
- name: home/seana/axolotl_project/outputs/phi2-vec-finetune
  results: []
---
> 🔍 This model (`phi2-vec-finetune`) is trained using `(dx, dy)` action vectors. Starting from a fixed origin (0,0), the model directly predicts the final position after applying all vectors. This version **does not use CoT-style explanation**, making it a lightweight baseline for reasoning ability.

Main use: baseline comparison for vector-only reasoning without step tracing or language interpretation.

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/axolotl-ai-cloud/axolotl)
<details><summary>See axolotl config</summary>

axolotl version: `0.7.0`
```yaml
base_model: /home/seana/axolotl_project/models/phi-2/phi2

attn_implementation: "eager"

model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer
trust_remote_code: true

special_tokens:
  pad_token: "<|endoftext|>"

datasets:
  - path: /home/seana/axolotl_project/data/simple_10x10_1to2000.jsonl
    type: alpaca

val_set_size: 0.05

adapter: lora
peft_type: LORA
lora_target_modules:
  - qkv_proj
  - out_proj
  - dense
  - dense_h_to_4h
  - dense_4h_to_h
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05

output_dir: /home/seana/axolotl_project/outputs/phi2-vec-finetune
train_on_inputs: false
group_by_length: false

bf16: false
fp16: false

load_in_4bit: false
load_in_8bit: true
bnb_8bit_compute_dtype: float16
use_bnb: true
llm_int8_enable_fp32_cpu_offload: true



optimizer: adamw_torch
micro_batch_size: 1
gradient_accumulation_steps: 4

num_train_epochs: 3
learning_rate: 2e-4
lr_scheduler_type: cosine
warmup_ratio: 0.05
weight_decay: 0.01

logging_steps: 10
eval_steps: 100
save_steps: 100
save_strategy: steps
save_total_limit: 2

report_to: none
seed: 42

```

</details><br>

# home/seana/axolotl_project/outputs/phi2-vec-finetune

This model was trained from scratch on the /home/seana/axolotl_project/data/simple_10x10_1to2000.jsonl dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0233

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 4
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 23
- num_epochs: 1.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| No log        | 0.0021 | 1    | 2.8055          |
| 0.1912        | 0.2105 | 100  | 0.2012          |
| 0.1134        | 0.4211 | 200  | 0.0800          |
| 0.0716        | 0.6316 | 300  | 0.0850          |
| 0.0228        | 0.8421 | 400  | 0.0233          |


### Framework versions

- PEFT 0.14.0
- Transformers 4.48.3
- Pytorch 2.5.1+cu121
- Datasets 3.2.0
- Tokenizers 0.21.2