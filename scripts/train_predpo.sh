while (( "$#" )); do
  case "$1" in
    --gpus)
      gpus="$2"
      shift 2
      ;;
    --dataset)
      dataset="$2"
      shift 2
      ;;
    --sft_model_path)
      sft_model_path="$2"
      shift 2
      ;;
    --ref_model_path)
      ref_model_path="$2"
      shift 2
      ;;
    --template)
      template="$2"
      shift 2
      ;;
    --pref_beta)
      pref_beta="$2"
      shift 2
      ;;
    --bsz)
      bsz="$2"
      shift 2
      ;;
    --gradient_accumulation_steps)
      gradient_accumulation_steps="$2"
      shift 2
      ;;
    --lr)
      lr="$2"
      shift 2
      ;;
    --warmup_ratio)
      warmup_ratio="$2"
      shift 2
      ;;
    --epoch)
      epoch="$2"
      shift 2
      ;;
    --output_dir)
      output_dir="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done


gpus=${gpus:-0,1,2,3,4,5,6,7}
sft_model_path=${sft_model_path}
ref_model_path=${ref_model_path}
pref_beta=${pref_beta:-0.1}
template=${template}
bsz=${bsz:-1}
gradient_accumulation_steps=${gradient_accumulation_steps:-16}
lr=${lr:-6.0e-7}
warmup_ratio=${warmup_ratio:-0.06}
epoch=${epoch:-1.0}
output_dir=${output_dir:-experiments/$(date +"%Y%m%d_%H%M%S")}

CUDA_VISIBLE_DEVICES=${gpus} llamafactory-cli train \
    --model_name_or_path ${sft_model_path} \
    --stage dpo \
    --do_train \
    --finetuning_type full \
    --deepspeed scripts/deepspeed_config/ds_z2_config.json \
    --dataset ${dataset} \
    --template ${template} \
    --cutoff_len 4096 \
    --overwrite_cache \
    --preprocessing_num_workers 64 \
    --output_dir ${output_dir} \
    --logging_steps 1 \
    --plot_loss \
    --overwrite_output_dir \
    --per_device_train_batch_size ${bsz} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate ${lr} \
    --num_train_epochs ${epoch} \
    --lr_scheduler_type cosine \
    --warmup_ratio ${warmup_ratio} \
    --bf16 \
    --ddp_timeout 180000000 \
    --save_only_model \
    --pref_beta ${pref_beta} \
    --ref_model ${ref_model_path}
