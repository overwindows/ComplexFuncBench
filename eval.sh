export OPENAI_API_KEY=3929f1c3-e3b5-40a7-9422-0e4bdc876e69
export OPENAI_API_URL=https://api.sambanova.ai/v1

export SAMBANOVA_API_URL=https://api.sambanova.ai/v1
export SAMBANOVA_API_KEY=042ca35c-beaf-4f5b-8033-9170556e5251

# MODEL_NAME=/nvmedata/hf_checkpoints/Llama-3.3-70B-Instruct
# EXP_NAME=Llama-3.3-70B-Instruct-Vanilla
# EXP_NAME=Llama-3.3-70B-Instruct-SpecPrefill

# MODEL_NAME=Meta-Llama-3.3-70B-Instruct
# EXP_NAME=Meta-Llama-3.3-70B-Instruct

# MODEL_NAME=DeepSeek-V3-0324
# EXP_NAME=DeepSeek-V3-0324

MODEL_NAME=Meta-Llama-3.1-8B-Instruct
EXP_NAME=Meta-Llama-3.1-8B-Instruct

# python evaluation.py --model_name $MODEL_NAME --proc_num 4 --vllm_url http://127.0.0.1:8000/v1 --exp_name $EXP_NAME
python evaluation.py --model_name $MODEL_NAME --proc_num 1 --exp_name $EXP_NAME