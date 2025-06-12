# -*- coding: utf-8 -*-
import argparse
import datetime
import json
import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import get_context

from runner.claude_runner import ClaudeRunner
from runner.glm_runner import GLMAPIRunner, GLMRunner
from runner.gpt_runner import GPTRunner
from runner.llama_runner import LlamaRunner
from runner.mistral_runner import MistralRunner
from runner.qwen_runner import QwenRunner
from runner.response_runner import RespEvalRunner
from utils.logger import Logger
from utils.utils import *

MODEL_MAPPING = {
    "gpt-4o-2024-08-06": GPTRunner,
    "gpt-4-turbo-2024-04-09": GPTRunner,
    "claude-3-5-sonnet-20240620": ClaudeRunner,
    "claude-3-5-sonnet-20241022": ClaudeRunner,
    "claude-3-5-haiku-20241022": ClaudeRunner,
    "glm-4-9b-chat": GLMRunner,
    "glm-4-long": GLMAPIRunner,
    "Llama-3.1-70B": LlamaRunner,
    "Llama-3.1-8B": LlamaRunner,
    "Meta-Llama-3.1-405B-Instruct-FP8": LlamaRunner,
    "qwen2.5-7b-instruct": QwenRunner,
    "qwen2.5-72b-instruct": QwenRunner,
    "qwen2.5-7b-instruct": QwenRunner,
    "mistral-large-2407": MistralRunner,
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, default="logs/test.log")
    parser.add_argument("--input-file", type=str, default="data/ComplexFuncBench.jsonl")
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="The name of the model to be evaluated.",
    )
    parser.add_argument(
        "--evaluator-model-name",
        type=str,
        default="gpt-4o-mini",
        help="The model used for response evaluation.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("BASE_URL", None),
        help="The base URL for the model API.",
    )
    parser.add_argument(
        "--base-url-evaluator",
        type=str,
        default=os.environ.get("EVALUATOR_BASE_URL", None),
        help="The API key for the evaluator model API.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("API_KEY", None),
        help="The API key for the model API.",
    )
    parser.add_argument(
        "--api-key-evaluator",
        type=str,
        default=os.environ.get("EVALUATOR_API_KEY", None),
        help="The API key for the evaluator model API.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HF_TOKEN", None),
        help="The Hugging Face token for accessing models.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="The maximum number of worker processes to use.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Whether to use strict mode for function calls.",
    )
    parser.add_argument("--exp-name", type=str, default="full-1000")
    parser.add_argument("--vllm-url", type=str)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    if not args.hf_token:
        raise ValueError(
            "Hugging Face token is required. Please set the HF_TOKEN environment variable or provide it as an argument."
        )

    if args.base_url and not args.base_url.endswith("/v1"):
        args.base_url = f"{args.base_url}/v1"

    if args.base_url_evaluator and not args.base_url_evaluator.endswith("/v1"):
        args.base_url_evaluator = f"{args.base_url_evaluator}/v1"

    dir_name = os.path.dirname(__file__)
    os.makedirs(
        f"{dir_name}/logs/{datetime.date.today().strftime('%Y-%m-%d')}/{args.model_name}",
        exist_ok=True,
    )

    os.makedirs(
        f"{dir_name}/result/{args.model_name}/{args.exp_name}/logs", exist_ok=True
    )

    args.log_dir = f"{dir_name}/logs/{datetime.date.today().strftime('%Y-%m-%d')}/{args.model_name}/{args.exp_name}.log"
    args.output_dir = f"{dir_name}/result/{args.model_name}/{args.exp_name}.jsonl"
    args.log_dir = f"{dir_name}/result/{args.model_name}/{args.exp_name}/logs"
    return args


def process_example(data, args):
    log_dir = f"{args.log_dir}/{data['id']}.log"
    logger = Logger(f"evaluation_logger_{data['id']}", log_dir, logging.DEBUG)

    if args.model_name not in MODEL_MAPPING:
        logger.warning(f"Unknown runner for model {args.model_name}.")
        model = GPTRunner(
            model_name=args.model_name,
            logger=logger,
            api_key=args.api_key,
            base_url=args.base_url,
            hf_token=args.hf_token,
            strict=args.strict
        )
    else:
        model = MODEL_MAPPING[args.model_name](
            model_name=args.model_name,
            logger=logger,
            api_key=args.api_key,
            base_url=args.base_url,
            hf_token=args.hf_token,
        )

    resp_eval_model = RespEvalRunner(
        model_name=args.evaluator_model_name,
        logger=logger,
        api_key=args.api_key_evaluator,
        base_url=args.base_url_evaluator,
    )

    logger.info(f"Test Example {data['id']}")
    logger.info(f"Query: {data['conversations'][0]['content']}")

    turn_count, call_count = 0, 0
    for turn in data["conversations"]:
        if turn["role"] == "assistant" and "function_call" in turn:
            turn_count += 1
            call_count += len(turn["function_call"])

    convs, message, turn_id, correct_count = model.run(data)

    # API Error
    if isinstance(message, dict) and message["error_type"] == "unknown_error":
        return None

    real_turn_count = 0
    for turn in convs:
        if turn["role"] == "assistant" and "function_call" in turn:
            real_turn_count += 1

    if convs[-1]["role"] == "assistant" and "content" in convs[-1]:
        gen_response = convs[-1]["content"]
        resp_eval_result = resp_eval_model.run(data, gen_response)
    else:
        resp_eval_result = None

    logger.info(f"Message: {message}")
    logger.info(f"Success turn num = {turn_id}")
    logger.info("-" * 100)

    result = {
        "id": data["id"],
        "gen_convs": convs,
        "message": message,
        "count_dict": {
            "success_turn_num": turn_id,
            "total_turn_num": turn_count,
            "correct_call_num": correct_count,
            "total_call_num": call_count,
            "real_turn_num": real_turn_count,
        },
        "resp_eval": resp_eval_result,
    }

    return result


def main():
    args = get_args()
    test_data = load_json(args.input_file)
    if args.debug:
        test_data = random.sample(test_data, 1)

    if os.path.exists(args.output_dir):
        finished_data = load_json(args.output_dir)
        finised_ids = [d["id"] for d in finished_data]
    else:
        finised_ids = []
    test_data = [d for d in test_data if d["id"] not in finised_ids]

    ctx = get_context("spawn")  # avoids AF_UNIX by not using fork+Manager

    _process_example = partial(process_example, args=args)
    with ProcessPoolExecutor(max_workers=args.max_workers, mp_context=ctx) as pool:
        with open(args.output_dir, "w") as f:
            for result in pool.map(_process_example, test_data):
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
