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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SHORT_PROMPTS = [
    "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
    "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
    "Paddington has 40 more goats than Washington. If Washington has 140 goats, how many goats do they have in total?",
    "Ed has 2 dogs, 3 cats and twice as many fish as cats and dogs combined. How many pets does Ed have in total?",
    "In 3 years, Jayden will be half of Ernesto's age. If Ernesto is 11 years old, how many years old is Jayden now?",
    "Mel is three years younger than Katherine. When Katherine is two dozen years old, how old will Mel be in years?",
    "A store sells 20 packets of 100 grams of sugar every week. How many kilograms of sugar does it sell every week?",
    "Herbert is 10 years younger than Kris. If Kris is 24 years old now, how old will Herbert be next year?",
    "John writes 20 pages a day. How long will it take him to write 3 books that are 400 pages each?",
    "Hash has nine more than half as many toys as Bill has. If Bill has 60 toys, how many total toys do the boys have?",
    "The selling price of a bicycle that had sold for $220 last year was increased by 15%. What is the new price?",
    "If Stu has 9 books and Albert has 4 times as many books as Stu, how many books do Stu and Albert have in total?",
    "Cori is 3 years old today. In 5 years, she will be one-third the age of her aunt. How old is her aunt today?",
]

LONG_PROMPT_PAIRS = [
    {
        "prompt": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
        "context": "He writes each friend 3*2=<<3*2=6>>6 pages a week So he writes 6*2=<<6*2=12>>12 pages every week That means he writes 12*52=<<12*52=624>>624 pages a year #### 624"
    },
    {
        "prompt": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "context": "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. #### 10"
    },
        {
        "prompt": "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
        "context": "In one hour, there are 3 sets of 20 minutes. So, Joy can read 8 x 3 = <<8*3=24>>24 pages in an hour. It will take her 120/24 = <<120/24=5>>5 hours to read 120 pages. #### 5"
    },
    {
        "prompt": "Paddington has 40 more goats than Washington. If Washington has 140 goats, how many goats do they have in total?",
        "context": "If Washington has 140 goats, Washington has 140+40 = <<140+40=180>>180 goats. In total, they have 140+180 = <<140+180=320>>320 goats #### 320"
    },
    {
        "prompt": "Ed has 2 dogs, 3 cats and twice as many fish as cats and dogs combined. How many pets does Ed have in total?",
        "context": "If Ed has 2 dogs and 3 cats he has in total 2+3 = <<2+3=5>>5 pets that are not fish If Ed has twice as many cats and dogs combined he has 2*5 = <<2*5=10>>10 fish Therefore, in total Ed has 5+10 = <<5+10=15>>15 pets #### 15"
    },
    {
        "prompt": "In 3 years, Jayden will be half of Ernesto's age. If Ernesto is 11 years old, how many years old is Jayden now?",
        "context": "Ernesto = 11 + 3 = <<11+3=14>>14 Jayden = 14/2 = <<14/2=7>>7 in 3 years Now = 7 - 3 = <<7-3=4>>4 Jayden is 4 years old. #### 4"
    },
    {
        "prompt": "Mel is three years younger than Katherine. When Katherine is two dozen years old, how old will Mel be in years?",
        "context": "When Katherine is 2 dozen years old, she will be 2*12=<<2*12=24>>24 years old. If Mel is three years younger than Katherine, then when Katherine is 24 years old, Mel will be 24-3=<<24-3=21>>21 years old. #### 21"
    },
    {
        "prompt": "A store sells 20 packets of 100 grams of sugar every week. How many kilograms of sugar does it sell every week?",
        "context": "A total of 20 x 100 = <<20*100=2000>>2000 grams are sold every week. Since 1 kilogram is equal to 1000 grams, then 2000/1000 = <<2000/1000=2>>2 kilograms of sugar are sold every week. #### 2"
    },
    {
        "prompt": "Herbert is 10 years younger than Kris. If Kris is 24 years old now, how old will Herbert be next year?",
        "context": "Herbert is 24 - 10 = <<24-10=14>>14 years old now. Thus, Herbert will be 14 + 1 = <<14+1=15>>15 years old next year. #### 15"
    },
    {
        "prompt": "John writes 20 pages a day. How long will it take him to write 3 books that are 400 pages each?",
        "context": "He wants to write 3*400=<<3*400=1200>>1200 pages So it will take him 1200/20=<<1200/20=60>>60 days #### 60"
    },
    {
        "prompt": "Hash has nine more than half as many toys as Bill has. If Bill has 60 toys, how many total toys do the boys have?",
        "context": "First we need to know what half of Bills toys are, 60 toys / 2 = <<60/2=30>>30 toys. Hash has 9 toys + 30 toys = <<9+30=39>>39 toys. Together the boys have 60 toys + 39 toys = <<60+39=99>>99 toys. #### 99"
    },
    {
        "prompt": "The selling price of a bicycle that had sold for $220 last year was increased by 15%. What is the new price?",
        "context": "The price of the bicycle increased by $220 * 15/100 = $<<220*15/100=33>>33. Adding the increment price, the new price is $220 + $33 = $<<220+33=253>>253. #### 253"
    },
    {
        "prompt": "If Stu has 9 books and Albert has 4 times as many books as Stu, how many books do Stu and Albert have in total?",
        "context": "Albert has 4 * 9 books belonging to Stu = <<4*9=36>>36 books. So the pair have a combined total of 36 books belonging to Albert + 9 books belonging to Stu = <<36+9=45>>45 books. #### 45"
    },
    {
        "prompt": "Cori is 3 years old today. In 5 years, she will be one-third the age of her aunt. How old is her aunt today?",
        "context": "In 5 years, Cori will be 3 + 5 = <<3+5=8>>8 years old. In 5 years, Cori’s aunt will be 8 x 3 = <<8*3=24>>24 years old. Today, her aunt is 24 - 5 = <<24-5=19>>19 years old. #### 19"
    },
    {
        "prompt": "James drives 30 mph for half an hour and then twice as long for twice the speed. How far did he drive in total?",
        "context": "His first drive was 30*.5=<<30*.5=15>>15 miles The next leg was .5*2=<<.5*2=1>>1 hour The speed of the trip was 30*2=<<30*2=60>>60 mph So he drove 60*1=<<60=60>>60 miles So in total he drove 60+15=<<60+15=75>>75 miles #### 75"
    },
    {
        "prompt": "Jessica is six years older than Claire. In two years, Claire will be 20 years old. How old is Jessica now?",
        "context": "Claire's age now is 20 - 2 = <<20-2=18>>18 years old. Being 6 years older than Claire, Jessica is 18 + 6 = <<6+18=24>>24 years old. #### 24"
    },
    {
        "prompt": "A luxury bag costs $3000. A reseller wants to get a 15% profit. How much should she sell the bag?",
        "context": "The reseller wants to get $3000 x 15/100 = $<<3000*15/100=450>>450 profit. Thus, she needs to sell it for $3000 + $450 = $<<3000+450=3450>>3450. #### 3450"
    },
    {
        "prompt": "Diane bought twenty more apples than Cecile. If Cecile bought 15 apples, how many apples did they buy altogether?",
        "context": "Diane bought 15 + 20 = <<15+20=35>>35 apples. Therefore, they bought 15 + 35 = <<15+35=50>>50 apples altogether. #### 50"
    },
    {
        "prompt": "Jacob is 24 years now. His brother Tony is half Jacob's age. In 6 years how old will tony be?",
        "context": "Tony’s age now is 24 / 2 = <<24/2=12>>12 years old. In 6 years he will be 12 + 6 = <<12+6=18>>18 years old. #### 18"
    },
]

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
    percentiles = [50, 95, 99]
    latency_percentiles = [calculate_percentile(latencies, p) for p in percentiles]
    tps_percentiles = [calculate_percentile(tokens_per_second_list, p, reverse=True) for p in percentiles]
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
            "p50": latency_percentiles[0],
            "p95": latency_percentiles[1],
            "p99": latency_percentiles[2]
        },
        "tokens_per_second": {
            "average": avg_tokens_per_second,
            "p50": tps_percentiles[0],
            "p95": tps_percentiles[1],
            "p99": tps_percentiles[2]
        },
        "time_to_first_token(ttft)": {
            "average": avg_ttft,
            "p50": ttft_percentiles[0],
            "p95": ttft_percentiles[1],
            "p99": ttft_percentiles[2]
        },
        "time-per-output-token(tpot)": {
            "average": avg_tpot,
            "p50": tpot_percentiles[0],
            "p95": tpot_percentiles[1],
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
        "latency_p50": latency_percentiles[0],
        "latency_p95": latency_percentiles[1],
        "latency_p99": latency_percentiles[2],
        "tokens_per_second_average": avg_tokens_per_second,
        "tokens_per_second_p50": tps_percentiles[0],
        "tokens_per_second_p95": tps_percentiles[1],
        "tokens_per_second_p99": tps_percentiles[2],
        "time_to_first_token_average": avg_ttft,
        "time_to_first_token_p50": ttft_percentiles[0],
        "time_to_first_token_p95": ttft_percentiles[1],
        "time_to_first_token_p99": ttft_percentiles[2],
        "time_per_output_token_average": avg_tpot,
        "time_per_output_token_p50": tpot_percentiles[0],
        "time_per_output_token_p95": tpot_percentiles[1],
        "time_per_output_token_p99": tpot_percentiles[2]
    }

    # add new row data to the existing data
    if not df.empty:
        # ensure that the existing data has at least 35 columns
        for column in new_row.keys():
            if column not in df.columns:
                df[column] = pd.NA

        # update the existing rows from column 11 to column 35.
        last_index = len(df) - 1
        for column, value in new_row.items():
            df.at[last_index, column] = value
    else:
        # update the existing rows from column 11 to column 35
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

