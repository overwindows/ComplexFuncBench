export OPENAI_API_KEY=3929f1c3-e3b5-40a7-9422-0e4bdc876e69

# MODEL_NAME=/nvmedata/hf_checkpoints/Llama-3.3-70B-Instruct
# EXP_NAME=Llama-3.3-70B-Instruct-Vanilla
# EXP_NAME=Llama-3.3-70B-Instruct-SpecPrefill

MODEL_NAME=DeepSeek-R1-0528
EXP_NAME=DeepSeek-R1-0528

# python evaluation.py --model_name $MODEL_NAME --proc_num 4 --vllm_url http://127.0.0.1:8000/v1 --exp_name $EXP_NAME
python evaluation.py --model_name $MODEL_NAME --proc_num 4 --exp_name $EXP_NAME