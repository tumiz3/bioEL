#!/bin/bash

exp_tag="e6"
python ./finetune_2.py \
    --base_model '../../huozi/model_weight/' \
    --data_path './data/new_aier_data/instructions/all_train_instructions.json' \
    --output_dir './fine_tune_BenTsao-'$exp_tag \
    --prompt_template_name 'med_template' \
    --micro_batch_size 8 \
    --batch_size 8 \
    --wandb_run_name $exp_tag