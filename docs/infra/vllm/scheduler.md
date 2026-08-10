# vLLM Scheduler：Parallel Sampling 与 Beam Search

## 多输出调度解决的问题

普通解码让一个 prompt 对应一条输出序列；Parallel Sampling 和 Beam Search 都会让一个 prompt 同时派生多条序列。此时系统既要调度每条活跃序列，又要在 API 边界恢复“一个请求”的语义。

- Parallel Sampling 需要让 `n` 条 child request 独立解码，再按原始 request 聚合结果。
- Beam Search 需要反复扩展活跃 beam、比较累计分数，并把候选集裁剪回 `beam_width`。

两者都呈现 fan-out / fan-in，但控制状态不同：Parallel Sampling 用 `ParentRequest` 管理父子关系，Beam Search 用 `BeamSearchInstance` 管理搜索状态。

## Sequence Group

从调度视角看，一个逻辑请求可能对应多条正在运行的 sequence。调度器关心的是每条 sequence 当前需要 prefill 还是 decode、占用多少 KV Cache，以及是否已经结束；上层组件再负责把这些 sequence 重新聚合成用户看到的请求输出。

这层拆分让 batch 可以混合来自不同父请求的 child sequence，而不要求它们每一步都同步完成。

## Parallel Sampling

当 `SamplingParams.n > 1` 时，vLLM 为同一个 prompt 创建多个 child request。每个 child 使用独立的采样状态，因而可以得到不同 completion。

### ParentRequest 的职责

`ParentRequest` 不执行模型推理，主要维护请求级元数据与聚合状态：

- 保存父级 `request_id` 和原始 `sampling_params`。
- 通过 `_get_child_sampling_params()` 为 child request 生成参数。
- 用 `child_requests` 跟踪尚未结束的 child。
- 在非流式或 `FINAL_ONLY` 输出模式下，用 `output_aggregator` 汇总 completion。
- 通过 `observe_num_generation_tokens()` 合并请求级统计信息。

因此，scheduler 看到的是可独立推进的 child request；API 层看到的仍是同一个父请求。

### 执行流程

1. 请求进入 engine 后创建一个 `ParentRequest`。
2. engine 调用 `get_child_info()` 生成 child `request_id` 与 child sampling params。
3. scheduler 把每个 child 当作独立请求参与 prefill、decode 和 KV Cache 管理。
4. child 产生 `RequestOutput` 后，输出处理层调用 `parent_req.get_outputs()`。
5. `get_outputs()` 根据输出模式立即转发单个 child，或等待全部 child 完成后统一聚合。
6. `child_requests` 为空后，父请求可以释放。

流式模式中的事件可以交错到达：

```text
RequestOutput(request_id="reqA", outputs=[index=1: "Hello"], finished=False)
RequestOutput(request_id="reqA", outputs=[index=0: "Hi"],    finished=False)
RequestOutput(request_id="reqA", outputs=[index=2: "Hey"],   finished=False)
...
RequestOutput(request_id="reqA", outputs=[index=2: "Hey there"], finished=True)
```

在 `RequestOutputKind.FINAL_ONLY` 模式下，聚合器只在所有 child 完成后返回一次：

```python
RequestOutput(
    request_id="reqA",
    outputs=[
        CompletionOutput(index=0, text="..."),
        CompletionOutput(index=1, text="..."),
        CompletionOutput(index=2, text="..."),
    ],
    finished=True,
)
```

这里的 `index` 是恢复稳定输出顺序的关键；child 的实际完成顺序不需要与 `index` 一致。

## Beam Search

Beam Search 不是简单地并行生成 `n` 个互不相关的样本。它在每一轮共享同一组搜索候选，用累计 log probability 选择最有希望的路径。

### BeamSearchInstance

`BeamSearchInstance` 是一个 prompt 的搜索上下文，主要包含：

- `beams`：仍可继续扩展的 `BeamSearchSequence`。
- `completed`：已经遇到终止条件的 sequence。

每个 `BeamSearchSequence` 保存 token、逐 token logprobs 和与请求有关的附加信息。搜索结束时，再从 `completed` 与剩余活跃候选中选出最终结果。

### 每轮扩展流程

1. 把 prompt 包装为只有一条初始 beam 的 `BeamSearchInstance`。
2. `_beam_search_step()` 收集每个 instance 的活跃 `beams`。
3. 为每条活跃 beam 发起一次只生成一个 token 的请求，并要求返回足够多的候选 logprobs。
4. 把每条 beam 与它的候选 token 组合，形成下一轮候选集。
5. 遇到 EOS 或停止条件的候选移入 `completed`。
6. 对未完成候选计算 beam score，并排序。
7. 只保留前 `beam_width` 条活跃 beam，其他候选立即丢弃。
8. 达到停止条件后，从完成与活跃集合中选出最终结果。

裁剪必须在每一轮扩展后立即发生。否则候选数量会随词表大小呈指数增长，KV Cache 与计算成本都无法控制。

## 两种策略对比

| 维度 | Parallel Sampling | Beam Search |
| --- | --- | --- |
| child 关系 | 相互独立 | 每轮来自共同候选集 |
| 选择依据 | 各自随机状态 | 累计 beam score |
| 核心状态 | `ParentRequest.child_requests` | `BeamSearchInstance.beams` |
| 聚合时机 | 流式转发或全部完成 | 每轮裁剪，结束后排序 |
| 主要目标 | 获得多样的多个 completion | 搜索高分序列 |

## 关键结论

- scheduler 负责推进 sequence，上层对象负责恢复 request 语义。
- `ParentRequest` 是 Parallel Sampling 的聚合器，不参与 token 计算。
- child request 可以交错完成，`index` 用来恢复稳定的 completion 顺序。
- `BeamSearchInstance` 同时维护活跃与完成候选，并在每轮把规模限制在 `beam_width`。
- Parallel Sampling 的多条路径彼此独立；Beam Search 的路径持续竞争并被裁剪。

## 参考资料

- [vLLM Parallel Sampling API 与源码](https://docs.vllm.ai/en/latest/api/vllm/v1/engine/parallel_sampling/)
- [vLLM Beam Search API 与源码](https://docs.vllm.ai/en/latest/api/vllm/beam_search/)
