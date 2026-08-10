# nano-vLLM 核心机制与项目架构

## Continuous Batching

Continuous Batching（连续批处理）让调度器在每轮 GPU 推理前重新决定本轮处理哪些请求。

### Stage 1：不做批处理

请求被顺序处理。decode 阶段每步只计算一个 token，GPU 利用率较低。

### Stage 2：Static Batching

同一个 batch 中的请求共享一次 GPU 调用，但会遇到两个问题：

- 提前完成的请求（early-finished）会留下空位。
- 后到的请求（late-joining）必须等待当前 batch 结束。

### Stage 3：Continuous Batching

在每次 GPU 推理前，调度器按以下流程组织请求：

1. 新请求进入 `waiting` 队列。
2. 检查 `waiting` 队列中是否有可执行 prefill 的请求：
   - 计算 prompt，写入 KV Cache，并产出第一个 token。
   - prefill 完成后，请求进入 `running` 队列。
3. 如果本轮没有需要执行 prefill 的请求，则从 `running` 队列取出正在生成的序列，执行一次 decode。
4. 某个序列生成完毕后，立即将其从 `running` 队列移除，并释放它占用的 KV Cache 块。

## 项目架构

- `LLMEngine` 负责初始化组件并驱动生成循环。
- `Scheduler` 在 `waiting` 与 `running` 队列之间调度 `Sequence`。
- `BlockManager` 为序列分配、复用和回收 KV Cache 物理块。
- `ModelRunner` 准备 batch、执行模型并完成采样。
