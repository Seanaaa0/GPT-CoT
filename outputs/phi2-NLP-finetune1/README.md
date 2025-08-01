---
library_name: peft
tags:
- generated_from_trainer
datasets:
- /home/seana/axolotl_project/data/nlp_10x10_multi_entre_shuffled.jsonl
base_model: /home/seana/axolotl_project/models/phi-2/phi2
model-index:
- name: home/seana/axolotl_project/outputs/phi2-NLP-finetune1
  results: []
---
> 🔍 This model (`phi2-NLP-finetune1`) is trained using **natural language instructions** such as `up`, `down`, `left`, `right` instead of vectors. It is based on the CoT-5 adapter and uses the same five-entry setup.

Main use: compare vector vs NLP instruction formats in reasoning tasks.

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
  - path: /home/seana/axolotl_project/data/nlp_10x10_multi_entre_shuffled.jsonl
    type: alpaca

val_set_size: 0.05 # 沒有驗證集
adapter: lora
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - v_proj

sequence_len: 512
sample_packing: false
pad_to_sequence_len: true

output_dir: /home/seana/axolotl_project/outputs/phi2-NLP-finetune1

# 載入先前 LoRA 訓練好的 adapter 權重
adapter_path: outputs/phi2-CoT-finetune5/checkpoint-760

gradient_accumulation_steps: 2
micro_batch_size: 4
num_epochs: 8
optimizer: adamw_bnb_8bit

lr_scheduler: cosine
learning_rate: 2e-4
train_on_inputs: false
group_by_length: false

bf16: false
fp16: true
tf32: true

gradient_checkpointing: true
early_stopping_patience: 0
logging_steps: 10
save_steps: 50
special_tokens:
  pad_token: "<|endoftext|>"

```

</details><br>

# home/seana/axolotl_project/outputs/phi2-NLP-finetune1

This model was trained from scratch on the /home/seana/axolotl_project/data/nlp_10x10_multi_entre_shuffled.jsonl dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0000

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
- lr_scheduler_warmup_steps: 28
- num_epochs: 8.0
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 0.0015        | 1.0   | 119  | 0.0003          |
| 0.0009        | 2.0   | 238  | 0.0006          |
| 0.0012        | 3.0   | 357  | 0.0001          |
| 0.0002        | 4.0   | 476  | 0.0002          |
| 0.0001        | 5.0   | 595  | 0.0000          |
| 0.0001        | 6.0   | 714  | 0.0000          |
| 0.0001        | 7.0   | 833  | 0.0000          |
| 0.0001        | 8.0   | 952  | 0.0000          |


### Framework versions

- PEFT 0.14.0
- Transformers 4.48.3
- Pytorch 2.5.1+cu121
- Datasets 3.2.0
- Tokenizers 0.21.2