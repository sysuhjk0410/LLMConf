# LLMConf: Knowledge-Enhanced Configuration Optimization for Large Language Model Inference

## 💡 What is LLMConf?
LLMConf is a multi-parameter tuning method for LLMs. By leveraging knowledge-enhanced techniques, we identify tuning parameters and their value ranges, significantly reducing the search space for parameter combinations. To capture the impact of configuration parameters on inference performance, we use the automated machine learning tool TPOT to model the functional relationships between configuration parameters and each performance metric. Additionally, to optimize multiple performance metrics simultaneously and resolve conflicts in optimization directions, we implement a multi-objective optimization module based on the genetic algorithm.

The experimental results show that LLMConf significantly outperforms state-of-the-art methods, achieving an average performance improvement of **19.8%** on **16** metrics.

LLMConf demonstrates a strong transferability across diverse datasets, varying concurrency levels, and different LLM base models.


# 🚀 Performance snapshot
We evaluate the inference performance of LLMs from two aspects: latency and throughput. In terms of latency, we consider ***latency*** (the time taken to complete each request), ***TTFT***(time_to_first_token), ***TPOT***(time_per_output_token). For throughput, we measure ***TPS***(tokens_per_second). Our 16 optimized metrics include *latency_average*, *latency_p50*, *latency_p95*, *latency_p99*, *TPS_average*, *TPS\_p50*, *TPS_p95*, *TPS_p99*, *TTFT_average*, *TTFT_p50*, *TTFT_p95*, *TTFT_p99*, *TPOT_average*, *TPOT_p50*, *TPOT_p95*, and *TPOT_p99*. To present the experimental results more intuitively, we select *latency_average*, *latency*_p99, *TPS_average, *TTFT_average*, *TTFT_p99*, *TPOT_average*, and *TPOT_p99* for visual presentations.
![image](https://github.com/sysuhjk0410/image/blob/main/LLMConf_result.png)
