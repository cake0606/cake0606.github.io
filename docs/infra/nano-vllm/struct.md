## 核心机制
### Contiguous Batching
Contiguous Batching（连续批处理）
#### stage 1: 不做批处理
顺序处理请求， decode阶段每步只算一个token, gpu利用率低
#### stage 2: static batching
同一个batch请求共享一次gpu调用
> early-finished
> late-joining
#### stage 3: Contiguous Batching
在每一次GPU推理之前， 调度器决定这一步处理哪些请求
1. 新请求->waiting 队列
2. 检查waiting 队列中有无可以prefill的请求
    - 有的话， prefill：计算prompt， 写入kv cache， 产出第一个token
    - 请求进入running 队列
3. 如果没有需要prefill的请求， 从running队列拉出所有正在生成的队列， 做一次decode
4. 某个序列生成完， 立即从running移除， 释放kv cache块

## 项目架构

