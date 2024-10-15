# LLMConf: Knowledge-Enhanced Configuration Optimization for Large Language Model Inference

## 💡 What is LLMConf?
LLMConf is a multi-parameter tuning method for LLMs. By leveraging knowledge-enhanced techniques, we identify tuning parameters and their value ranges, significantly reducing the search space for parameter combinations. To capture the impact of configuration parameters on inference performance, we use the automated machine learning tool TPOT to model the functional relationships between configuration parameters and each performance metric. Additionally, to optimize multiple performance metrics simultaneously and resolve conflicts in optimization directions, we implement a multi-objective optimization module based on the genetic algorithm.

The experimental results show that LLMConf significantly outperforms state-of-the-art methods, achieving an average performance improvement of **_19.8%_** on **_16_** metrics.

LLMConf demonstrates a strong transferability across diverse datasets, varying concurrency levels, and different LLM base models.


# 🚀 Performance snapshot
We evaluate the inference performance of LLMs from two aspects: latency and throughput. In terms of latency, we consider ***_latency_*** (the time taken to complete each request)
![image](https://github.com/sysuhjk0410/image/blob/main/LLMConf_exp.png)
