# nano-vLLM ModelRunner

## 初始化

`ModelRunner.__init__` 先初始化 NCCL 进程组与当前 CUDA 设备，再加载模型。模型加载后依次调用 `warmup_model()`、`allocate_kv_cache()`；未启用 `enforce_eager` 时，还会调用 `capture_cudagraph()`。

## warmup_model

`warmup_model()` 构造受 `max_num_batched_tokens`、`max_model_len` 和 `max_num_seqs` 限制的最大规模假 prefill：

```python
def warmup_model(self):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    max_num_batched_tokens = self.config.max_num_batched_tokens
    max_model_len = self.config.max_model_len
    seq_len = min(max_num_batched_tokens, max_model_len)
    num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
    seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
    for seq in seqs:
        seq.num_scheduled_tokens = seq_len
    self.run(seqs, True)
    torch.cuda.empty_cache()
```

这一步有两个作用：

1. 预先触发模型首次执行的一次性开销：
   - CUDA kernel 首次加载。
   - `torch.compile` 或图优化相关初始化。
   - 算子 workspace 分配。
   - 通信库与 Attention backend 的懒初始化。
2. 记录更接近真实峰值的显存占用，避免高估可分配给 KV Cache 的空间。PyTorch 的 CUDA allocator 会在内存统计中保留历史分配峰值。

## allocate_kv_cache

`allocate_kv_cache()` 先根据显存预算计算可分配的物理块数量：

```text
available_bytes = total * gpu_memory_utilization - used - peak + current
num_kv_cache_blocks = available_bytes // block_bytes
```

计算出 `num_kv_cache_blocks` 后，一次性分配全局 KV Cache 张量：

```python
self.kv_cache = torch.empty(
    2,
    num_layers,
    num_kv_cache_blocks,
    block_size,
    num_kv_heads,
    head_dim,
)
```

各维含义如下：

- 第 0 维：K 和 V。
- 第 1 维：模型层。
- 第 2 维：物理块 ID。
- 第 3 维：块内 token 偏移。
- 第 4、5 维：该 token 的 KV 向量，即 KV head 和 head dimension。

随后按层绑定 `k_cache` 与 `v_cache`：

```python
layer_id = 0
for module in self.model.modules():
    if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
        module.k_cache = self.kv_cache[0, layer_id]
        module.v_cache = self.kv_cache[1, layer_id]
        layer_id += 1
```

每层缓存的形状是 `[num_kv_cache_blocks, block_size, num_kv_heads, head_dim]`。其中，一个物理块对应一页，块内 token 偏移对应页内地址。

## capture_cudagraph

`capture_cudagraph()` 先准备一组最大容量的静态缓冲区：

```python
input_ids = torch.zeros(max_bs, dtype=torch.int64)
positions = torch.zeros(max_bs, dtype=torch.int64)
slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
context_lens = torch.zeros(max_bs, dtype=torch.int32)
block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
outputs = torch.zeros(max_bs, hf_config.hidden_size)
```

CUDA Graph capture 要求 capture 时使用的张量地址稳定。replay 时只更新这些已有张量中的值，避免在 replay 路径中重新申请张量。

`graph_bs` 记录一系列 batch size，`capture_cudagraph()` 为每个大小捕获一张图；decode 时选择不小于实际 batch size 的最小图执行 replay。

## 将 Sequence 转成 GPU Batch

### Context

Prefill 与 decode 的输入形态不同。`Context` 用于在模型执行前注入本轮 batch 的 KV Cache 写入位置、历史上下文布局，以及 prefill 或 decode 所需的元数据。

### run

`run()` 一次处理一批 `Sequence`：

```python
def run(self, seqs, is_prefill):
    input_ids, positions = (
        self.prepare_prefill(seqs)
        if is_prefill
        else self.prepare_decode(seqs)
    )

    temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
    logits = self.run_model(input_ids, positions, is_prefill)
    token_ids = (
        self.sampler(logits, temperatures).tolist()
        if self.rank == 0
        else None
    )

    reset_context()
    return token_ids
```

### prepare_prefill

`prepare_prefill()` 只把本轮需要计算的 token 加入 `input_ids`，并记录它们在原序列中的绝对位置：

```python
for seq in seqs:
    start = seq.num_cached_tokens
    seqlen_q = seq.num_scheduled_tokens
    end = start + seqlen_q
    seqlen_k = end
    input_ids.extend(seq[start:end])
    positions.extend(range(start, end))
    cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
    cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
    max_seqlen_q = max(seqlen_q, max_seqlen_q)
    max_seqlen_k = max(seqlen_k, max_seqlen_k)
```

`seqlen_q` 是本轮要计算的 token 数；`seqlen_k` 是本轮可见的 K/V 长度。`cu_seqlens_q` 与 `cu_seqlens_k` 是 cumulative sequence lengths，用于记录 batch 展平后的序列边界。

假设两条未命中 Prefix Cache 的序列长度分别为 11 和 17，且 `block_size = 256`，展平结果如下：

```text
input_ids = [t0, t1, ..., t10, t0, t1, ..., t16]  # 长度 28
positions = [0, 1, ..., 10, 0, 1, ..., 16]         # 长度 28

cu_seqlens_q = [0, 11, 28]
cu_seqlens_k = [0, 11, 28]

seq_id=4: block_table=[0]
seq_id=5: block_table=[1]
slot_mapping = [0, 1, ..., 10, 256, 257, ..., 272]
```

`slot_mapping` 的构造过程会用 `seq.block_table[i]` 将逻辑块映射到物理块，并保留本轮每个 token 的槽位：

```python
start_block = start // self.block_size
end_block = (end + self.block_size - 1) // self.block_size
for i in range(start_block, end_block):
    slot_start = seq.block_table[i] * self.block_size
    if i == start_block:
        slot_start += start % self.block_size
    if i != end_block - 1:
        slot_end = seq.block_table[i] * self.block_size + self.block_size
    else:
        slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
    slot_mapping.extend(range(slot_start, slot_end))
```

### prepare_decode

`prepare_decode()` 处理上一轮采样得到的新 token，为本轮 forward 写入该 token 的 KV 提前准备地址：

```python
slot = (
    seq.block_table[-1] * self.block_size
    + seq.last_block_num_tokens
    - 1
)
```

本轮 decode forward 会把该 token 的 KV 写入这个槽位，并预测下一个 token。
