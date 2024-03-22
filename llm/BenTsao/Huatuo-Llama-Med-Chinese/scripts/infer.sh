#!/bin/sh

# If inferring with the llama model, set 'use_lora' to 'False' and 'prompt_template' to 'ori_template'.
# If inferring with the default alpaca model, set 'use_lora' to 'True', 'lora_weights' to 'tloen/alpaca-lora-7b', and 'prompt_template' to 'alpaca'.
# If inferring with the llama-med model, download the LORA weights and set 'lora_weights' to './lora-llama-med' (or the exact directory of LORA weights) and 'prompt_template' to 'med_template'.

python infer2.py \
    --base_model '../../huozi/model_weight/' \
    --lora_weights1 '../Lora' \
    --lora_weights2 './fine_tune_BenTsao-e6/checkpoint-64' \
    --use_lora True \
    --instruct_dir './data/new_aier_data/instructions/another_test_instructions.json' \
    --prompt_template 'med_template'
