# LLMConf: Knowledge-Enhanced Configuration Optimization for Large Language Model Inference (IWQoS 2025)

## 💡 What is LLMConf?
LLMConf is a multi-parameter tuning method for LLMs. By leveraging knowledge-enhanced techniques, we identify tuning parameters and their value ranges, significantly reducing the search space for parameter combinations. To capture the impact of configuration parameters on inference performance, we use the automated machine learning tool TPOT to model the functional relationships between configuration parameters and each performance metric. Additionally, to optimize multiple performance metrics simultaneously and resolve conflicts in optimization directions, we implement a multi-objective optimization module based on the NSGA-III algorithm.

The experimental results show that LLMConf significantly outperforms state-of-the-art methods, achieving an average performance improvement of **20.1%** on **7** metrics.

LLMConf demonstrates a strong transferability across diverse datasets, varying concurrency levels and different LLM base models.

![overview of LLMConf](https://github.com/sysuhjk0410/LLMConf/blob/main/workflow.png) 

# 🚀 Performance snapshot
We evaluate the inference performance of LLMs from two aspects: latency and throughput. In terms of latency, we consider ***latency*** (the time taken to complete each request), time to first token(***TTFT***) and time per output token(***TPOT***). For throughput, we measure tokens per second(***TPS***). Our 7 optimized metrics include *latency_average*, *latency_p99*, *TPS_average*, *TTFT_average*, *TTFT_p99*, *TPOT_average* and *TPOT_p99*.

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
For the prompt used in determining the range and type of each tuning parameters, refer to `TypeandRange.txt` (using the determination of the value range for `max-num-batched-tokens` as an example). Run the following command to complete the determination of the range and type of each tuning parameter, setting the `file_path` value to `./TypeandRange.txt`.
```bash
python llm_chat.py
```
✨️ Note: The `api_key` and `base_url` need to be filled in.

## - Data Collector
Run the following command to deploy LLM (the `BaseLLLM` folder needs to be created before downloading LLM).
```bash
modelscope download --model 'LLM-Research/Meta-Llama-3-8B-Instruct' --local_dir 'BaseLLM/Meta-Llama-3-8B-Instruct'
modelscope download --model 'Qwen/Qwen2.5-14B-Instruct' --local_dir 'BaseLLM/Qwen/Qwen2.5-14B-Instruct'
```
Run the following command to automate data collection.
```bash
python auto.py
```
✨️ Note:
- `config.yml`: Include the range and type of all tuning parameters.
- `SetConfig.py`: Randomly set the values of various configuration parameters.
- `vllm_benchmark.py`: Test the inference performance of the LLM by a series of performance metrics.

If only collecting the inference performance of the LLM under a specific combination of configuration parameters, the following command can be run.
```bash
vllm serve /LLMConf/BaseLLM/Meta-Llama-3-8B-Instruct --port 8100
python /LLMConf/vllm_benchmark1.py --num_requests 200 --concurrency 80 --output_tokens 200 --vllm_url http://localhost:8100/v1 --api_key EMPTY
```
✨️ If using the `ChatDoctor-HealthCareMagic-100k` dataset for testing, `vllm_benchmark_Health.py` can be run.

✨️ Note: All the data collected in this experiment can be found in the `Data` folder.(The naming of the CSV file sequentially represents LLM, concurrency and dataset.)

## - Performance Modeling
First, move the data that needs to be modeled to the `Modeling` directory.
Run the following command to convert the Boolean type in the data to 0 or 1.
```bash
cd Modeling
python convert.py
```
Run the following command to complete the modeling of the tuning parameters with each performance metric.
```bash
python modeling.py
```
⚖️ Other modeling methods:
- `CNN model`:
```bash
cd comp
python CNN.py
```
- `MLP model`:
```bash
cd comp
python MLP.py
```
- `Random Forest model`:
```bash
cd comp
python RandomForest.py
```
- `SVM model`:
```bash
cd comp
python SVM.py
```
- `XGBoost model`:
```bash
cd comp
python XGBoost.py
```

## - Multi-Objective Optimization
Run the following command to complete the recommendation of optimal configuration parameters.
```bash
cd Functions
python optimize.py
```
⚖️ Other multi-objective optimization algorithms:
- `RS`:
```bash
python RS.py
```
- `SCOOT`:
```bash
python SCOOT.py
```
- `MAB`:
```bash
python MultiBandit.py
```
- `DDPG`:
```bash
python DDPG.py
```
- `NSGA-III`:
```bash
python NSGA3.py
```
