import asyncio
import time
import numpy as np
import pandas as pd
from openai import AsyncOpenAI
import logging
import argparse
import json
import random
import csv
import os
from datasets import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load HealthCareMagic-100k dataset
HealthCareMagic_dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", "main")["train"]

SHORT_PROMPTS = [example["question"] for example in random.sample(list(HealthCareMagic_dataset), 15)]

LONG_PROMPT_PAIRS = []
selected_examples = random.sample(list(HealthCareMagic_dataset), 32)
for example in selected_examples:
    answer = example["answer"]
    context = answer.split("####")[0].strip()
    LONG_PROMPT_PAIRS.append({
        "prompt": example["question"],
        "context": context
    })

async def process_stream(stream):
    stream_message = ''
    first_token_time = None
    total_tokens = 0
    async for chunk in stream:
        if first_token_time is None:
            first_token_time = time.time()
        if chunk.choices[0].delta.content:
            total_tokens += 1
            stream_message += chunk.choices[0].delta.content
        if chunk.choices[0].finish_reason is not None:
            break
    logging.info(stream_message)
    logging.info(f'total tokens={total_tokens}')
    return first_token_time, total_tokens

async def make_request(client, output_tokens, request_timeout, use_long_context):
    start_time = time.time()
    if use_long_context:
        prompt_pair = random.choice(LONG_PROMPT_PAIRS)
        content = prompt_pair["context"] + "\n\n" + prompt_pair["prompt"]
    else:
        content = random.choice(SHORT_PROMPTS)
    logging.info(content)
    try:
        stream = await client.chat.completions.create(
            model="/LLMConf/BaseLLM/Meta-Llama-3-8B-Instruct",
            messages=[{"role": "user", "content": content}],
            max_tokens=output_tokens,
            stream=True
        )
        
        first_token_time, total_tokens = await asyncio.wait_for(process_stream(stream), timeout=request_timeout)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        ttft = first_token_time - start_time if first_token_time else None
        tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0
        return total_tokens, elapsed_time, tokens_per_second, ttft

    except asyncio.TimeoutError:
        logging.warning(f"Request timed out after {request_timeout} seconds")
        return None
    except Exception as e:
        logging.error(f"Error during request: {str(e)}")
        return None

async def worker(client, semaphore, queue, results, output_tokens, request_timeout, use_long_context):
    while True:
        async with semaphore:
            task_id = await queue.get()
            if task_id is None:
                queue.task_done()
                break
            logging.info(f"Starting request {task_id}")
            result = await make_request(client, output_tokens, request_timeout, use_long_context)
            if result:
                results.append(result)
            else:
                logging.warning(f"Request {task_id} failed")
            queue.task_done()
            logging.info(f"Finished request {task_id}")

def calculate_percentile(values, percentile, reverse=False):
    if not values:
        return None
    if reverse:
        return np.percentile(values, 100 - percentile)
    return np.percentile(values, percentile)

async def run_benchmark(num_requests, concurrency, request_timeout, output_tokens, vllm_url, api_key, use_long_context):
    client = AsyncOpenAI(base_url=vllm_url, api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)
    queue = asyncio.Queue()
    results = []

    # add tasks to the queue
    for i in range(num_requests):
        await queue.put(i)
    
    # add termination signals to stop worker threads
    for _ in range(concurrency):
        await queue.put(None)

    # create worker threads
    workers = [asyncio.create_task(worker(client, semaphore, queue, results, output_tokens, request_timeout, use_long_context)) for _ in range(concurrency)]

    start_time = time.time()
    
    # wait for all tasks to complete
    await queue.join()
    await asyncio.gather(*workers)

    end_time = time.time()

    # calculate metrics
    total_elapsed_time = end_time - start_time
    total_tokens = sum(tokens for tokens, _, _, _ in results if tokens is not None)
    latencies = [elapsed_time for _, elapsed_time, _, _ in results if elapsed_time is not None]
    tokens_per_second_list = [tps for _, _, tps, _ in results if tps is not None]
    ttft_list = [ttft for _, _, _, ttft in results if ttft is not None]
    tpot_list = [x / y for x, y in zip(latencies, [tokens for tokens, _, _, _ in results if tokens is not None])]

    successful_requests = len(results)
    requests_per_second = successful_requests / total_elapsed_time if total_elapsed_time > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    avg_tokens_per_second = sum(tokens_per_second_list) / len(tokens_per_second_list) if tokens_per_second_list else 0
    avg_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else 0
    avg_tpot = sum(tpot_list) / len(tpot_list) if tpot_list else 0

    # calculate percentiles
    percentiles = [99]
    latency_percentiles = [calculate_percentile(latencies, p) for p in percentiles]
    ttft_percentiles = [calculate_percentile(ttft_list, p) for p in percentiles]
    tpot_percentiles = [calculate_percentile(tpot_list, p) for p in percentiles]

    results_dict = {
        "total_requests": num_requests,
        "successful_requests": successful_requests,
        "concurrency": concurrency,
        "request_timeout": request_timeout,
        "max_output_tokens": output_tokens,
        "use_long_context": use_long_context,
        "total_time": total_elapsed_time,
        "requests_per_second": requests_per_second,
        "total_output_tokens": total_tokens,
        "latency": {
            "average": avg_latency,
            "p99": latency_percentiles[2]
        },
        "tokens_per_second": {
            "average": avg_tokens_per_second
        },
        "time_to_first_token(ttft)": {
            "average": avg_ttft,
            "p99": ttft_percentiles[2]
        },
        "time-per-output-token(tpot)": {
            "average": avg_tpot,
            "p99": tpot_percentiles[2]
        }
    }

    # write results to CSV file
    csv_file = '/LLMConf/data/data.csv'
    file_exists = os.path.isfile(csv_file)
    
    # read existing data
    if file_exists:
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame()

    # create new row data
    new_row = {
        "latency_average": avg_latency,
        "latency_p99": latency_percentiles[2],
        "TPS_average": avg_tokens_per_second,
        "TTFT_average": avg_ttft,
        "TTFT_p99": ttft_percentiles[2],
        "TPOT_average": avg_tpot,
        "TPOT_p99": tpot_percentiles[2]
    }

    # add new row data to the existing data
    if not df.empty:
        # ensure that the existing data has all required columns
        for column in new_row.keys():
            if column not in df.columns:
                df[column] = pd.NA
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    # save the updated data to the CSV file
    df.to_csv(csv_file, index=False)

    return results_dict

def print_results(results):
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark LLaMA-3 model with vLLM")
    parser.add_argument("--num_requests", type=int, required=True, help="Number of requests to make")
    parser.add_argument("--concurrency", type=int, required=True, help="Number of concurrent requests")
    parser.add_argument("--request_timeout", type=int, default=30, help="Timeout for each request in seconds (default: 30)")
    parser.add_argument("--output_tokens", type=int, default=50, help="Number of output tokens (default: 50)")
    parser.add_argument("--vllm_url", type=str, required=True, help="URL of the vLLM server")
    parser.add_argument("--api_key", type=str, required=True, help="API key for vLLM server")
    parser.add_argument("--use_long_context", action="store_true", help="Use long context prompt pairs instead of short prompts")
    args = parser.parse_args()

    results = asyncio.run(run_benchmark(args.num_requests, args.concurrency, args.request_timeout, args.output_tokens, args.vllm_url, args.api_key, args.use_long_context))
    print_results(results)
else:
    # When imported as a module, provide the run_benchmark function
    __all__ = ['run_benchmark']
