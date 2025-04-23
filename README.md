# Pre-DPO

## Overview

This is the repository for the paper [Pre-DPO: Improving Data Utilization in Direct Preference Optimization Using a Guiding Reference Model](https://arxiv.org/abs/2504.15843).

Pre-DPO is a simple yet effective DPO-based training paradigm that enhances preference optimization performance by leveraging a **guiding reference model**.

This repository is based on the popular repository [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), which can easily fine-tune 100+ large language models.


## Installation

First, create a new conda environment and activate it.

```shell
conda create -n predpo python=3.10 && conda activate predpo
```

Next, clone the repository and install PyTorch along with the remaining dependencies.

```shell
git clone https://github.com/DtYXs/Pre-DPO.git
cd Pre-DPO
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[torch,metrics]"
pip install deepspeed==0.15.4
```

## Training

### Training Data

For the Base models ([Llama3.2-3B-Base](https://huggingface.co/meta-llama/Llama-3.2-3B) and [Qwen2.5-7B-Base](https://huggingface.co/Qwen/Qwen2.5-7B)), we utilize the [UltraChat-200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) dataset to obtain the SFT models. Subsequently, we perform preference optimization using the [UltraFeedback-Binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) dataset.

For the Instruct models ([Llama3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) and [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)), we follow the [pipeline](https://github.com/princeton-nlp/SimPO/tree/main/on_policy_data_gen) described in SimPO to generate on-policy preference data, using [ArmoRM-Llama3-8B-v0.1](https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1) as the preference label annotator. The resulting preference datasets are [llama3.2-3b-ultrafeedback-armorm-binarized](https://huggingface.co/datasets/DtYXs/llama3.2-3b-ultrafeedback-armorm-binarized) and [qwen2.5-7b-ultrafeedback-armorm-binarized](https://huggingface.co/datasets/DtYXs/qwen2.5-7b-ultrafeedback-armorm-binarized), respectively.

You can refer `./data/README.md` and prepare your data in `./data/dataset_info.json`.

### Training Scripts

We provide our training scripts and examples in `./scripts`. We train the 3B models on 4 × 80G GPUs and the 7B models on 8 × 80G GPUs.

#### SFT
```shell
bash scripts/train_sft.sh --model_name_or_path <MODEL_NAME_OR_PATH> --dataset <DATASET_NAME> --output_dir <OUTPUT_DIR> --template <TEMPLATE>
```

#### DPO
```shell
bash scripts/train_dpo.sh --sft_model_path <SFT_MODEL_PATH> --dataset <DATASET_NAME> --output_dir <OUTPUT_DIR> --template <TEMPLATE> --pref_beta <BETA_IN_DPO> --bsz <BATCH_SIZE> --gradient_accumulation_steps <GRADIENT_ACCUMULATION_STEPS> --lr <LEARNING_RATE>
```

#### SimPO
```shell
bash scripts/train_simpo.sh --sft_model_path <SFT_MODEL_PATH> --dataset <DATASET_NAME> --output_dir <OUTPUT_DIR> --template <TEMPLATE> --pref_beta <BETA_IN_SIMPO> --simpo_gamma <GAMMA_IN_SIMPO> --bsz <BATCH_SIZE> --gradient_accumulation_steps <GRADIENT_ACCUMULATION_STEPS> --lr <LEARNING_RATE>
```

#### Pre-DPO
```shell
bash scripts/train_predpo.sh --sft_model_path <SFT_MODEL_PATH> --ref_model_path <REF_MODEL_PATH> --dataset <DATASET_NAME> --output_dir <OUTPUT_DIR> --template <TEMPLATE> --pref_beta <BETA_IN_PREDPO> --bsz <BATCH_SIZE> --gradient_accumulation_steps <GRADIENT_ACCUMULATION_STEPS> --lr <LEARNING_RATE>
```

# Evaluation

We conduct evaluations on AlpacaEval 2.0 and Arena-Hard v0.1 following their official repositories.

+ [AlpacaEval repository](https://github.com/tatsu-lab/alpaca_eval)
+ [Arena-Hard repository](https://github.com/lmarena/arena-hard-auto)

# Acknowledgement

We deeply appreciate the outstanding open-source code of [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) and [SimPO](https://github.com/princeton-nlp/SimPO), which has greatly supported research efforts within the community.

# Citiation

if Pre-DPO is helpful to your work, please cite our paper:
```bibtex
@misc{pan2025predpoimprovingdatautilization,
      title={Pre-DPO: Improving Data Utilization in Direct Preference Optimization Using a Guiding Reference Model}, 
      author={Junshu Pan and Wei Shen and Shulin Huang and Qiji Zhou and Yue Zhang},
      year={2025},
      eprint={2504.15843},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.15843}, 
}
```

# Contact

+ Email: panjunshu@westlake.edu.cn
