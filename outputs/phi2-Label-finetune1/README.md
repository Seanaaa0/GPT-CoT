---
library_name: peft
tags:
- generated_from_trainer
datasets:
- /home/seana/axolotl_project/data/10x10_vec_labeled.jsonl
base_model: /home/seana/axolotl_project/models/phi-2/phi2
model-index:
- name: home/seana/axolotl_project/outputs/phi2-Label-finetune1
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/axolotl-ai-cloud/axolotl)
<details><summary>See axolotl config</summary>

axolotl version: `0.7.0`
```yaml
base_model: /home/seana/axolotl_project/models/phi-2/phi2
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer

load_in_4bit: true
strict: false

datasets:
  - path: /home/seana/axolotl_project/data/10x10_vec_labeled.jsonl
    type: alpaca

val_set_size: 0.05 
adapter: lora
lora_r: 8
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - v_proj

sequence_len: 512
sample_packing: false
pad_to_sequence_len: true

output_dir: /home/seana/axolotl_project/outputs/phi2-Label-finetune1

# 載入先前 LoRA 訓練好的 adapter 權重
adapter_path: /home/seana/axolotl_project/outputs/phi2-CoT-finetune5/checkpoint-760

gradient_accumulation_steps: 2
micro_batch_size: 4
num_epochs: 5
optimizer: adamw_bnb_8bit

lr_scheduler: cosine
learning_rate: 2e-4
train_on_inputs: false
group_by_length: true
num_virtual_tokens: 32

bf16: false
fp16: true
tf32: true

gradient_checkpointing: true

eval_steps: 10

early_stopping_patience: 5
logging_steps: 10
save_steps: 50
special_tokens:
  pad_token: "<|endoftext|>"

```

</details><br>

# home/seana/axolotl_project/outputs/phi2-Label-finetune1

This model was trained from scratch on the /home/seana/axolotl_project/data/10x10_vec_labeled.jsonl dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0035

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
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- gradient_accumulation_steps: 2
- total_train_batch_size: 8
- optimizer: Use adamw_bnb_8bit with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 10
- num_epochs: 5.0
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| No log        | 0.0140 | 1    | 0.3703          |
| 0.2705        | 0.1399 | 10   | 0.1777          |
| 0.1263        | 0.2797 | 20   | 0.0097          |
| 0.0092        | 0.4196 | 30   | 0.0234          |
| 0.0147        | 0.5594 | 40   | 0.0133          |
| 0.0018        | 0.6993 | 50   | 0.0486          |
| 0.0286        | 0.8392 | 60   | 0.0092          |
| 0.0083        | 0.9790 | 70   | 0.0126          |
| 0.0072        | 1.1119 | 80   | 0.0036          |
| 0.0068        | 1.2517 | 90   | 0.0018          |
| 0.0018        | 1.3916 | 100  | 0.0024          |
| 0.0042        | 1.5315 | 110  | 0.0005          |
| 0.0004        | 1.6713 | 120  | 0.0132          |
| 0.0052        | 1.8112 | 130  | 0.0059          |
| 0.0004        | 1.9510 | 140  | 0.0353          |
| 0.009         | 2.0839 | 150  | 0.0009          |
| 0.0011        | 2.2238 | 160  | 0.0035          |


### Framework versions

- PEFT 0.14.0
- Transformers 4.48.3
- Pytorch 2.5.1+cu121
- Datasets 3.2.0
- Tokenizers 0.21.2