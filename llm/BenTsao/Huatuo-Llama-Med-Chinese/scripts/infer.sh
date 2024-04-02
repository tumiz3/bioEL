#!/bin/sh

# If inferring with the llama model, set 'use_lora' to 'False' and 'prompt_template' to 'ori_template'.
# If inferring with the default alpaca model, set 'use_lora' to 'True', 'lora_weights' to 'tloen/alpaca-lora-7b', and 'prompt_template' to 'alpaca'.
# If inferring with the llama-med model, download the LORA weights and set 'lora_weights' to './lora-llama-med' (or the exact directory of LORA weights) and 'prompt_template' to 'med_template'.

python infer2.py \
    --base_model '../../huozi/model_weight/' \
    --lora_weights1 '../Lora' \
    --lora_weights2 '../../../distillation_from_gpt3.5_to_BenTsao/train-e1/checkpoint-864' \
    --use_lora True \
    --instruct_dir '../../../distillation_from_gpt3.5_to_BenTsao/data/test_instructions.json' \
    --prompt_template 'med_template'
