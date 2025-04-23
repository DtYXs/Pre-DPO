bash scripts/train_predpo.sh \
    --gpus 0,1,2,3 \
    --sft_model_path meta-llama/Llama-3.2-3B-Instruct \
    --ref_model_path experiments/Llama3.2-3B-Instruct-DPO \
    --dataset llama3.2-3b-ultrafeedback-armorm-binarized \
    --template llama3 \
    --pref_beta 0.05 \
    --lr 1e-6 \
    --gradient_accumulation_steps 32 \
    --output_dir experiments/Llama3.2-3B-Instruct-PreDPO-DPO
