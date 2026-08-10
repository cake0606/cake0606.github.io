# nano-vLLM Scheduler

## Sequence 状态

`Sequence` 是一条请求的状态机。

### 字段

| 字段 | 含义 |
| --- | --- |
| `seq_id` | 全局唯一 ID，自增分配 |
| `token_ids` | 当前完整序列，即 prompt 与已生成 token 的组合 |
| `last_token` | 最后一个 token；decode 时只需要计算这一个 token 的 Q |
| `num_tokens` | 当前总长度 |
| `num_prompt_tokens` | 不变的 prompt 长度 |
| `num_cached_tokens` | 前缀中命中 Prefix Cache 的 token 数 |
| `block_table` | 该序列占用的物理 KV Cache 块 ID 列表 |
| `status` | `WAITING`、`RUNNING` 或 `FINISHED` |
| `temperature`、`max_tokens`、`ignore_eos` | 从 `SamplingParams` 复制的采样参数 |

## BlockManager 接口

| 方法 | 作用 | 调用时机 |
| --- | --- | --- |
| `can_allocate(seq)` | 判断空闲块能否容纳序列当前长度，并返回可复用的完整前缀块数 | prefill 准入判断 |
| `allocate(seq, num_cached_blocks)` | 填充 `seq.block_table`，并更新 `seq.num_cached_tokens` | 从 `waiting` 队列取出请求时 |
| `can_append(seq)` | 判断下一步 decode 是否需要新块，以及空闲块是否足够 | decode 准入判断 |
| `may_append(seq)` | 在跨块时分配新块 | 请求进入 decode batch 前 |
| `deallocate(seq)` | 减少序列所引用块的 `ref_count`，计数归零时回收块 | 序列结束或被 `preempt` 时 |

## `schedule`

### Prefill 阶段

`schedule` 先计算请求本轮还需要 prefill 多少 token：

- 如果请求还没有 `block_table`，调用 `can_allocate(seq)` 估算：
  - prompt 开头有多少个完整块可以直接复用 Prefix Cache。
  - 这些块对应多少已经缓存的 token。
  - 本轮待计算数量为 `seq.num_tokens - num_cached_blocks * block_size`。
- 如果请求已经有 `block_table`，说明它不是第一次调度，可能执行过部分 chunked prefill，或正在继续补齐剩余 prompt。此时待计算数量为 `seq.num_tokens - seq.num_cached_tokens`。

```python
if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
    break
```

这项判断只允许本轮第一个请求执行 chunked prefill；一旦 batch 中已有请求，后续无法完整放入的请求要等到下一轮。

### Decode 阶段

1. 从 `running` 队首取出一个请求。
2. 调用 `can_append(seq)` 判断是否有空间容纳下一步 decode：
   - 如果空间足够，将请求加入本轮 batch：

     ```python
     seq.num_scheduled_tokens = 1
     seq.is_prefill = False
     self.block_manager.may_append(seq)
     scheduled_seqs.append(seq)
     ```

   - 如果空间不足，不断从 `running` 队尾抢占其他活跃请求；如果已经没有其他请求，则抢占当前请求本身。
3. 调整成功调度的请求顺序，再放回 `running` 队列。

被抢占的请求会由 `preempt(seq)` 重置为 `WAITING`，释放其 KV Cache 块，并回到 `waiting` 队首。

## `postprocess`

`postprocess` 对本轮执行过的每个 `seq` 依次完成以下工作：

1. 调用 `hash_blocks(seq)`，把新形成的完整块注册到 Prefix Cache。
2. 增加 `seq.num_cached_tokens`，并清零 `seq.num_scheduled_tokens`。
3. 如果本轮仍是未完成的 prefill，只更新进度，不产生输出 token。
4. 否则，将采样得到的 `token_id` 追加到序列。
5. 如果遇到 EOS，或 `seq.num_completion_tokens == seq.max_tokens`：
   - 将状态标记为 `FINISHED`。
   - 释放 KV Cache。
   - 从 `running` 队列移除该序列。
