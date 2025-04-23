bash scripts/train_dpo.sh \
    --gpus 0,1,2,3 \
    --sft_model_path meta-llama/Llama-3.2-3B-Instruct \
    --dataset llama3.2-3b-ultrafeedback-armorm-binarized \
    --template llama3 \
    --pref_beta 0.05 \
    --lr 6e-7 \
    --gradient_accumulation_steps 32 \
    --output_dir experiments/Llama3.2-3B-Instruct-DPO
