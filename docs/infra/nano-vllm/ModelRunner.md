---
id: ModelRunner
aliases: []
tags: []
---

## init

## warmup
```python
def warmup_model(self):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # 构造最大规模的假 prefill：max_num_batched_tokens / max_model_len 条序列，每条 max_model_len 长
    num_seqs = min(max_num_batched_tokens // max_model_len, max_num_seqs)
    seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
    self.run(seqs, True)  # 跑一次 prefill
    torch.cuda.empty_cache()
```
1. 预先触发模型首次执行的各种一次性开销
- CUDA kernel 首次加载
- torch.compile/图优化相关初始化
- 算子 workspace 分配
- 通信库、attention backend 的懒初始化
2. 避免高估可用显存
    torch的CUDA allocator会记录历史分配peak。

## allocate_kv_cache
```
available_bytes = total * gpu_memory_utilization - used - peak + current
num_kv_cache_blocks = available_bytes // block_bytes
```

计算出块数之后， 分配一个tensor
```python
self.kv_cache = torch.empty(
    2,
    num_layers,
    num_blocks,
    block_size,
    num_kv_heads,
    head_dim
)
```

第 0 维：2，分别是 K 和 V
第 1 维：layer
第 2 维：block id
第 3 维：block 内 token 偏移
第 4/5 维：这个 token 的 KV 向量

给每一层绑定kvcache
```python
for module in self.model.modules():
    if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
        module.k_cache = self.kv_cache[0, layer_id]
        module.v_cache = self.kv_cache[1, layer_id]
        layer_id += 1
```

其中各层的kvcache是[num_blocks, block_size, num_kv_heads, head_dim]
block对应page
token对应address

## capture_graph
1. 准备一组最大容量静态缓冲区
```python
input_ids = torch.zeros(max_bs, dtype=torch.int64)
positions = torch.zeros(max_bs, dtype=torch.int64)
slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
context_lens = torch.zeros(max_bs, dtype=torch.int32)
block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
outputs = torch.zeros(max_bs, hf_config.hidden_size)
```
CUDA Graph capture 要求：
capture 时用到的 tensor 地址稳定
replay 时最好只是“改这些已有 tensor 里的值”
不要在 replay 路径里重新申请一堆新 tensor

2. 为一系列batch size录一张图

## 把一批sequence变成gpu上的batch
### Context 全局上下文
prefill和decode输入形态不同， 利用 Context 把“本轮 batch 的 cache 写入位置、历史上下文布局、prefill/decode 元信息”提前注入到执行环境里。

### run（）
run一次处理一批序列
```python
def run(self, seqs, is_prefill):
    # 1. 准备输入：根据 prefill/decode 拼不同形态的 tensor + set_context
    input_ids, positions = self.prepare_prefill(seqs) if is_prefill \
                           else self.prepare_decode(seqs)

    # 2. 准备采样温度（仅 rank 0 需要）
    temperatures = self.prepare_sample(seqs) if self.rank == 0 else None

    # 3. 跑模型：得到 logits
    logits = self.run_model(input_ids, positions, is_prefill)

    # 4. 采样：logits + temperature → token_ids（仅 rank 0）
    token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None

    # 5. 清理上下文
    reset_context()
    return token_ids
```

### prepare_prefill
```python
:
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
seqlen_q代表本轮要计算的token数。seqlen_k代表本轮可见的k v长度
cu_seqlens： cumulative seqlens。记录的是batch拼接后的边界
> ```python
    # 展平 input_ids：所有 token 拼成一维
input_ids = [t0,t1,...,t10, t0,t1,...,t16]  # 长度 28

# positions：每个 token 的绝对位置（RoPE 用）
positions = [0,1,...,10, 0,1,...,16]  # 长度 28

# cu_seqlens_q：累积长度，标记序列边界
cu_seqlens_q = [0, 11, 28]  # seq_id=4 占 [0,11)，seq_id=5 占 [11,28)

# cu_seqlens_k：K 侧累积长度（无 prefix cache 时和 Q 相同）
cu_seqlens_k = [0, 11, 28]

# slot_mapping：每个 token 写入 KV cache 的物理位置
# seq_id=4 的 block_table=[0]，所以 slot 是 0*256+0, 0*256+1, ..., 0*256+10
# seq_id=5 的 block_table=[1]，所以 slot 是 1*256+0, 1*256+1, ..., 1*256+16
slot_mapping = [0,1,...,10, 256,257,...,272]  # 长度 28
```

```python

            for i in range(start_block, end_block):
                slot_start = seq.block_tableecho mapleader[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
```
计算token在物理kv cache中应该写到哪个slot
通过`seq.block_table[i]`来映射到对应的物理块, 在mapping中记录本轮token的slot位置

### prepare decode
prepare_decode 处理的是上一轮得出的新 token。为“这一轮 forward 写这个 token 的 KV”提前准备地址。然后这一轮 decode 前向真正完成 KV 写入和 next token 预测

