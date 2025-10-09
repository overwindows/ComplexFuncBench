# -*- coding: utf-8 -*-
import json
import random
import argparse
import os
import logging
import datetime
from collections import defaultdict
import multiprocessing
from multiprocessing import Pool, Manager
from functools import partial

from utils.logger import Logger
from utils.utils import *

from runner.gpt_runner import GPTRunner
from runner.glm_runner import GLMRunner, GLMAPIRunner
from runner.claude_runner import ClaudeRunner
from runner.qwen_runner import QwenRunner
from runner.qwen_runner_sn import QwenRunnerSN
from runner.llama_runner import LlamaRunner
from runner.mistral_runner import MistralRunner
from runner.response_runner import RespEvalRunner
from runner.deepseek_runner import DeepSeekRunner

MODEL_MAPPING = {
    "gpt-4o-2024-08-06": GPTRunner,
    "gpt-4-turbo-2024-04-09": GPTRunner,
    "gpt-oss-120b": GPTRunner,
    "claude-3-5-sonnet-20241022": ClaudeRunner,
    "claude-3-5-haiku-20241022": ClaudeRunner,
    "glm-4-9b-chat": GLMRunner,
    "glm-4-long": GLMAPIRunner,
    "Llama-3.1-70B": LlamaRunner,
    "Meta-Llama-3.1-8B-Instruct": LlamaRunner,
    "Meta-Llama-3.3-70B-Instruct": LlamaRunner,
    "Meta-Llama-3.1-405B-Instruct-FP8": LlamaRunner,
    "qwen2.5-7b-instruct": QwenRunner,
    "qwen2.5-72b-instruct": QwenRunner,
    "qwen2.5-7b-instruct": QwenRunner,
    "Qwen/Qwen3-32B": QwenRunner,
    "Qwen3-32B": QwenRunnerSN,
    "mistral-large-2407": MistralRunner,
    "DeepSeek-V3-0324": DeepSeekRunner,
    "DeepSeek-R1-0528": DeepSeekRunner,
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs/test.log")
    parser.add_argument("--input_file", type=str,
                        default="data/ComplexFuncBench.jsonl")
    parser.add_argument("--model_name", type=str, required=True, choices=list(
        MODEL_MAPPING.keys()), help="The name of the model to be evaluated.")
    parser.add_argument('--exp_name', type=str, default='full-1000')
    parser.add_argument("--vllm_url", type=str)
    parser.add_argument("--proc_num", type=int, default=1)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    os.makedirs(
        f"logs/{datetime.date.today().strftime('%Y-%m-%d')}/{args.model_name}", exist_ok=True)
    os.makedirs(
        f"result/{args.model_name}/{args.exp_name}/logs", exist_ok=True)

    args.log_dir = f"logs/{datetime.date.today().strftime('%Y-%m-%d')}/{args.model_name}/{args.exp_name}.log"
    args.output_dir = f"result/{args.model_name}/{args.exp_name}.jsonl"
    args.log_dir = f"result/{args.model_name}/{args.exp_name}/logs"
    return args


def process_example(data, args):
    log_dir = f"{args.log_dir}/{data['id']}.log"
    logger = Logger(f"evaluation_logger_{data['id']}", log_dir, logging.DEBUG)

    model = MODEL_MAPPING[args.model_name](args=args, logger=logger)
    resp_eval_model = RespEvalRunner(args=args, logger=logger)

    logger.info(f"Test Example {data['id']}")
    # logger.info(f"Query: {data['conversations'][0]['content']}")

    turn_count, call_count = 0, 0
    for turn in data['conversations']:
        if turn['role'] == "assistant" and "function_call" in turn:
            turn_count += 1
            call_count += len(turn["function_call"])

    convs, message, turn_id, correct_count = model.run(data)

    # API Error
    if isinstance(message, dict) and message["error_type"] == "unknown_error":
        return None

    real_turn_count = 0
    for turn in convs:
        if turn['role'] == "assistant" and "function_call" in turn:
            real_turn_count += 1

    # Skip response evaluation if no OpenAI API key is available
    assert os.getenv(
        "OPENAI_API_KEY"), "Please provide a valid OpenAI API key."
    # assert convs[-1]['role'] == "assistant" and "content" in convs[-1], f"convs[-1]: {convs[-1]}"
    # assert "content" in convs[-1], "convs[-1]: {convs[-1]}"
    if os.getenv("OPENAI_API_KEY") and convs[-1]['role'] == "assistant" and "content" in convs[-1]:
        gen_response = convs[-1]['content']
        resp_eval_result = resp_eval_model.run(data, gen_response)
    else:
        logger.warning(
            f"No OpenAI API key available or no generated response found. convs[-1]: {convs[-1]}")
        resp_eval_result = None

    logger.info(f"Message: {message}")
    logger.info(f"Success turn num = {turn_id}")
    logger.info("-" * 100)

    result = {
        "id": data['id'],
        "gen_convs": convs,
        "message": message,
        "count_dict": {
            "success_turn_num": turn_id,
            "total_turn_num": turn_count,
            "correct_call_num": correct_count,
            "total_call_num": call_count,
            "real_turn_num": real_turn_count
        },
        "resp_eval": resp_eval_result
    }

    with open(args.output_dir, 'a+') as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        f.flush()

    return result


def main():
    args = get_args()
    test_data = load_json(args.input_file)
    if args.debug:
        test_data = random.sample(test_data, 10)

    if os.path.exists(args.output_dir):
        finished_data = load_json(args.output_dir)
        finised_ids = [d["id"] for d in finished_data]
    else:
        finised_ids = []
    test_data = [d for d in test_data if d['id'] not in finised_ids]
    assert args.proc_num == 1
    # Use single processing to avoid multiprocessing issues with FlagEmbedding
    if args.proc_num == 1:
        results = []
        for data in test_data:
            result = process_example(data, args)
            results.append(result)
    else:
        # Try multiprocessing first, fallback to single processing if it fails
        try:
            with Manager() as manager:
                pool = Pool(processes=args.proc_num)
                process_example_partial = partial(process_example)
                results = pool.starmap(process_example_partial, [
                                       (data, args) for data in test_data])

            pool.close()
            pool.join()
        except Exception as e:
            print(
                f"Multiprocessing failed: {e}. Falling back to single processing.")
            results = []
            for data in test_data:
                result = process_example(data, args)
                results.append(result)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
