import yaml
import subprocess
import random
import time
import os
from SetConfig import get_command

for i in range(1500):  # Assume you need to execute 1500 times
    print(f"Starting iteration {i + 1}")

    # Set the environment variable CUDA_VISIBLE_DEVICES=5
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "5"

    # command = get_command()
    x = get_command()
    split_strings = x.split(' ')
    yy = [f"{s}" for s in split_strings]
    print(yy)


    # Start a subprocess and specify dynamic port
    process = subprocess.Popen(
        yy,
        env=env
    )

    # Check if vllm has started, wait a maximum of 60 seconds
    start_time = time.time()
    while process.poll() is None and (time.time() - start_time) < 60:
        time.sleep(5)
        print('Checking if vllm is ready...')
    
    if process.poll() is not None:
        print(f"vllm failed to start in iteration {i + 1}")
        continue  # Skip this iteration and continue to the next

    print('vllm ready')
    n=50 #Size specified by the user
    # Start the b.py process, connecting to the correct port
    process_b = subprocess.Popen(
        ["python", "/LLMConf/vllm_benchmark.py", "--num_requests", "200", "--concurrency", "50", 
        "--output_tokens","200", "--vllm_url", f"http://localhost:8100/v1", "--api_key", "EMPTY"]
    )

    # Wait for b.py to finish running
    process_b.wait()

    # Let vllm run for a few more seconds
    time.sleep(30)

    # Interrupt the vllm process
    print(f"Terminating vllm in iteration {i + 1}")
    process.terminate()  # Send SIGTERM signal to vllm to gracefully terminate the process

    # If vllm does not respond, use kill() to force terminate
    try:
        process.wait(timeout=3)  # Wait 3 seconds for vllm to exit
    except subprocess.TimeoutExpired:
        print("vllm did not exit gracefully, killing the process")
        process.kill()  # Force terminate vllm

    print(f"Finished iteration {i + 1}\n, wait for 10s ")
    time.sleep(10)
