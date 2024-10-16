# LLMConf: Knowledge-Enhanced Configuration Optimization for Large Language Model Inference

## 💡 What is LLMConf?
LLMConf is a multi-parameter tuning method for LLMs. By leveraging knowledge-enhanced techniques, we identify tuning parameters and their value ranges, significantly reducing the search space for parameter combinations. To capture the impact of configuration parameters on inference performance, we use the automated machine learning tool TPOT to model the functional relationships between configuration parameters and each performance metric. Additionally, to optimize multiple performance metrics simultaneously and resolve conflicts in optimization directions, we implement a multi-objective optimization module based on the genetic algorithm.

The experimental results show that LLMConf significantly outperforms state-of-the-art methods, achieving an average performance improvement of **19.8%** on **16** metrics.

LLMConf demonstrates a strong transferability across diverse datasets, varying concurrency levels, and different LLM base models.

![overview of LLMConf](https://github.com/sysuhjk0410/LLMConf/blob/main/workflow.png) 

# 🚀 Performance snapshot
We evaluate the inference performance of LLMs from two aspects: latency and throughput. In terms of latency, we consider ***latency*** (the time taken to complete each request), time to first token(***TTFT***), time per output token(***TPOT***). For throughput, we measure tokens per second(***TPS***). Our 16 optimized metrics include *latency_average*, *latency_p50*, *latency_p95*, *latency_p99*, *TPS_average*, *TPS_p50*, *TPS_p95*, *TPS_p99*, *TTFT_average*, *TTFT_p50*, *TTFT_p95*, *TTFT_p99*, *TPOT_average*, *TPOT_p50*, *TPOT_p95*, and *TPOT_p99*. To present the experimental results more intuitively, we select *latency_average*, *latency*_p99, *TPS_average, *TTFT_average*, *TTFT_p99*, *TPOT_average*, and *TPOT_p99* for visual presentations.

From the figure below, it can be seen that the optimization results of LLMConf are noticeably superior to those of other multi-objective optimization algorithms.
![experiment result](https://github.com/sysuhjk0410/LLMConf/blob/main/exp.png) 

# 💻 Quickstart

**Set Up Python Environment:** Use the following commands to create and activate the python environment:
```bash
conda create -n LLMConf python=3.10
conda activate LLMConf
```

**Install Dependencies:** install the necessary dependencies by running:
```bash
pip install -r requirements.txt
```

After completing the above steps, move into the `LLMConf` directory and follow the steps below to run the LLMConf project.

## - Knowledge-based Parameter Selection
We need to structure the constructed knowledge base into the prompt. 

For the prompt used in parameter selection, refer to `SelectConfiguration.txt`. Run the following command to complete the tuning parameter selection, setting the `file_path` value to `./SelectConfiguration.txt`.
```bash
cd LLMConf
python llm_chat.py
```
For the prompt used in determining the range and type of each tuning parameters, refer to `TypeandRange.txt`. Run the following command to complete the determination of the range and type of each tuning parameter, setting the `file_path` value to `./TypeandRange.txt`.
```bash
python llm_chat.py
```
✨️ Note: The `api_key` and `base_url` need to be filled in.

## - Data Collector
Run the following command to deploy LLM (the `BaseLLLM` folder needs to be created before downloading LLM).
```bash
export VLLM_USE_MODELSCOPE=True
modelscope download --model 'LLM-Research/Meta-Llama-3-8B-Instruct' --local_dir 'BaseLLM/Meta-Llama-3-8B-Instruct'
modelscope download --model 'Qwen/Qwen2.5-14B-Instruct' --local_dir 'BaseLLM/Qwen/Qwen2.5-14B-Instruct'
```
Run the following command to automate data collection.
```bash
python auto.py
```

