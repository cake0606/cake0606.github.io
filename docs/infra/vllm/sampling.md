# vLLM Sampling

## Greedy 与 Random

模型前向计算会为词表中的每个 token 产生一个 `logit`，采样器需要据此选出下一个 `token_id`。vLLM 的单步选择可以先分成两类：

- Greedy Sampling 直接选择 `argmax(logits)`，相同输入会得到确定性的下一个 token。
- Random Sampling 把处理后的 logits 转成概率分布，再按概率随机抽取 token。内部的 `RANDOM_SEED` 仍属于 Random，只是使用固定 seed 提供可复现的随机序列。

Random Sampling 还可以在抽样前调整或缩小候选集合：`temperature` 控制分布的平滑程度，Top-k 只保留分数最高的 `k` 个 token，Top-p 保留累计概率达到阈值的最小候选集合，Min-p 则按最高概率的相对比例过滤低概率 token。它们不是与 Greedy、Random 并列的采样类型，而是 Random 路径上的分布调整与候选过滤步骤。

Top-k、Top-p 和 Min-p 不会影响 Greedy 的最终结果。纯 Greedy batch 会在这些步骤之前返回；混合 batch 即使为向量化计算执行了 Random 路径，Greedy 行最后仍会选回预先计算的 `argmax`。

下面的流程图从用户输入的 `temperature` 开始，展示参数归一化、`SamplingType` 判定，以及纯 Greedy、纯 Random 和混合 batch 的执行路径。

![vLLM Greedy 与 Random Sampling 流程](../../assets/infra/vllm/greedy-random-sampling-flow.svg?v=20260811-2)

### 参数归一化与类型判定

`SamplingParams.sampling_type` 用归一化后的 `temperature` 与 `_SAMPLING_EPS` 判断采样类型，而不是直接读取未经处理的用户输入：

```python
@cached_property
def sampling_type(self) -> SamplingType:
    if self.temperature < _SAMPLING_EPS:
        return SamplingType.GREEDY
    if self.seed is not None:
        return SamplingType.RANDOM_SEED
    return SamplingType.RANDOM
```

当前源码中的两个阈值分别是：

```python
_SAMPLING_EPS = 1e-5
_MAX_TEMP = 1e-2
```

关键在于两个判断的执行顺序。`SamplingParams.__post_init__` 先抬高过小的正温度，完成参数校验后，才执行 Greedy 归一化：

```python
if 0 < self.temperature < _MAX_TEMP:
    self.temperature = max(self.temperature, _MAX_TEMP)

self._verify_args()

if self.temperature < _SAMPLING_EPS:
    self.top_p = 1.0
    self.top_k = 0
    self.min_p = 0.0
    self._verify_greedy_sampling()
```

因此，按用户的原始输入划分，行为是：

1. 原始输入 `temperature < 0`：在 `_verify_args()` 中被拒绝。
2. 原始输入 `temperature = 0`：不会触发 `_MAX_TEMP` 钳位，随后满足 `_SAMPLING_EPS` 条件，进入 `SamplingType.GREEDY`。
3. 原始输入 `0 < temperature < 0.01`：先被钳位到 `_MAX_TEMP = 0.01`，随后进入 Random。也就是说，原始输入 `1e-6` 不会进入 Greedy。
4. 原始输入 `temperature >= 0.01`：保留给定温度并进入 Random。

进入 Greedy 分支后，`top_p = 1.0`、`top_k = 0` 和 `min_p = 0.0` 都表示不截断候选 token；`_verify_greedy_sampling()` 还要求请求的 `n` 必须为 `1`。

### Batch 聚合与执行路径

请求加入 batch 时，元数据层会记录每个请求的采样类型。Greedy 请求的温度槽位保持为 `0.0`，Random 请求则保存真实温度：

```python
if sampling_params.sampling_type == SamplingType.GREEDY:
    self.temperature_cpu[req_index] = 0.0
    self.greedy_reqs.add(req_id)
else:
    self.temperature_cpu[req_index] = sampling_params.temperature
    self.random_reqs.add(req_id)
```

随后可以从集合状态派生 `all_greedy` 与 `all_random`。`Sampler.sample()` 根据这两个 batch 级标记选择快路径，而不需要为每一行启动不同的 kernel。

#### All Greedy

当 `all_greedy` 为真时，不需要把 temperature tensor 复制到 GPU。采样器直接计算 `argmax` 并提前返回：

```python
greedy_sampled = self.greedy_sample(logits)
if sampling_metadata.all_greedy:
    return greedy_sampled, processed_logprobs
```

这条路径不做温度缩放，不执行 Top-k、Top-p 或 Min-p，也不调用随机数生成器。

#### All Random

当 `all_random` 为真时，采样器跳过预先计算的 Greedy `argmax`，依次执行温度缩放、argmax-invariant processors、Top-k/Top-p 截断和随机抽样。

#### 混合 Batch

混合 batch 需要同时保留 Greedy 与 Random 两种结果。采样器先对整批 `logits` 计算 `greedy_sampled`，再生成 `random_sampled`。为避免 Greedy 行执行除零，`apply_temperature()` 用 `torch.where` 临时把这些行的温度替换为 `1.0`：

```python
@staticmethod
def apply_temperature(logits, temp, all_random):
    if not all_random:
        temp = torch.where(temp < _SAMPLING_EPS, 1.0, temp)
    return logits.div_(temp.unsqueeze(dim=1))
```

最后再次使用请求级 temperature mask 合并两条路径：

```python
sampled = torch.where(
    sampling_metadata.temperature < _SAMPLING_EPS,
    greedy_sampled,
    random_sampled,
    out=greedy_sampled,
)
```

因此，Greedy 行虽然参与了 Random 路径的批量 tensor 运算，最终结果仍取自预先计算的 `argmax`。

### Logits 处理顺序

`Sampler` 并非拿到原始 `logits` 就立即选择 token。以当前 V1 实现为例，主要顺序是：

1. 应用 allowed-token mask、bad words 等候选约束。
2. 应用可能改变 `argmax` 的 processors，例如最小生成长度与 logit bias。
3. 应用 repetition、frequency 和 presence penalties。
4. 如果 batch 含 Greedy 请求，先计算 `argmax`；若 `all_greedy` 为真则提前返回。
5. 对 Random 路径应用 `temperature`。
6. 应用不改变 `argmax` 的 processors，例如默认的 Min-p processor。
7. 应用 Top-k/Top-p 并执行随机抽样。
8. 混合 batch 用 `torch.where` 选回每一行对应的结果。

“argmax-invariant”表示 processor 不会改变最高分 token 的身份，所以它可以放在 Greedy 结果计算之后；可能改变最高分 token 的 processor 则必须先执行。

### Random 候选过滤

Top-k 和 Top-p 都发生在 Random 路径中。它们可以单独使用，也可以组合使用；过滤后的 logits 再被转换为概率分布并参与随机抽样。

#### Top-k

Top-k 保留分数最高的 `k` 个 token，其余位置写入负无穷：

```python
top_k_mask = logits_sort.size(1) - k.to(torch.long)
top_k_threshold = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
logits_sort.masked_fill_(logits_sort < top_k_threshold, -float("inf"))
```

当 `top_k = 0` 或 `top_k = -1` 时不限制候选集合。虽然 `top_k = 1` 在没有其他改变排序的处理时通常会得到与 Greedy 相同的 token，但只要归一化后的 temperature 大于 `_SAMPLING_EPS`，vLLM 内部仍按 Random 路径执行。

#### Top-p

Top-p 按概率质量保留最小候选集合。vLLM 的实现可以在升序排列上屏蔽累计概率位于 `1 - p` 之前的低概率 token，并强制至少保留一个候选：

```python
probs_sort = logits_sort.softmax(dim=-1)
probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
top_p_mask[:, -1] = False
logits_sort.masked_fill_(top_p_mask, -float("inf"))
```

处理完毕后，`scatter_()` 按 `logits_idx` 把排序后的分数写回原 token 顺序。`top_p = 1.0` 表示保留完整候选集合。

## 多序列生成

Greedy 与 Random 描述每一步怎样选择 token；Parallel Sampling 与 Beam Search 则描述一个 prompt 怎样维护多条输出序列。两者都会形成 fan-out / fan-in：底层调度每条活跃 sequence，上层对象再恢复“一个请求”的输出语义。

### Parallel Sampling

当 `SamplingParams.n > 1` 时，vLLM 为同一个 prompt 创建多个 child request。每个 child 独立执行 Random Sampling，因此可以产生不同的 completion；它们之间不会比较累计分数或互相裁剪。

#### ParentRequest 的职责

`ParentRequest` 不执行模型推理，主要维护父子关系与聚合状态：

- 保存父级 `request_id` 和原始 `sampling_params`。
- 通过 `_get_child_sampling_params()` 为 child request 生成参数。
- 用 `child_requests` 跟踪尚未结束的 child。
- 在非流式或 `FINAL_ONLY` 输出模式下，用 `output_aggregator` 汇总 completion。
- 通过 `observe_num_generation_tokens()` 合并请求级统计信息。

如果父请求设置了 seed，每个 child 会获得不同的 seed；否则 child 可以共享只读的采样参数。Scheduler 看到的是可独立推进的 child request，API 层看到的仍是同一个父请求。

#### 执行流程

1. 请求进入 engine 后创建一个 `ParentRequest`。
2. Engine 调用 `get_child_info()` 生成 child `request_id` 与 child sampling params。
3. Scheduler 把每个 child 当作独立请求参与 prefill、decode 和 KV Cache 管理。
4. Child 产生 `RequestOutput` 后，输出处理层调用 `parent_req.get_outputs()`。
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

这里的 `index` 用来恢复稳定的输出顺序；child 的实际完成顺序不需要与 `index` 一致。

### Beam Search

Beam Search 不是简单地并行生成多个互不相关的样本。它在每一轮共享同一组搜索候选，用累计 log probability 选择最有希望的路径，并把活跃候选数量限制在 `beam_width`。

#### BeamSearchInstance

`BeamSearchInstance` 是一个 prompt 的搜索上下文，主要包含：

- `beams`：仍可继续扩展的 `BeamSearchSequence`。
- `completed`：已经遇到终止条件的 sequence。

每个 `BeamSearchSequence` 保存 token、逐 token logprobs、累计 log probability 和请求附加信息。搜索结束时，再从 `completed` 与剩余活跃候选中选出最终结果。

#### 每轮扩展流程

1. 把 prompt 包装为只有一条初始 beam 的 `BeamSearchInstance`。
2. `_beam_search_step()` 收集每个 instance 的活跃 `beams`。
3. 为每条活跃 beam 发起一次只生成一个 token 的请求，并要求返回足够多的候选 logprobs。
4. 把每条 beam 与它的候选 token 组合，形成下一轮候选集。
5. 遇到 EOS 或停止条件的候选移入 `completed`。
6. 对未完成候选计算 beam score，并按累计分数排序。
7. 只保留前 `beam_width` 条活跃 beam，其他候选立即丢弃。
8. 达到停止条件后，从完成与活跃集合中选出最终结果。

裁剪必须在每一轮扩展后立即发生，否则候选数量会随词表大小快速增长，KV Cache 与计算成本都无法控制。
