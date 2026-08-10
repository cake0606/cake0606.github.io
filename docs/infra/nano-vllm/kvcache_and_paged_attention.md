---
id: kvcache_and_paged_attention
aliases: []
tags: []
---

# nano-vllm的kvcache和paged attention
## 架构
// 把这一段格式变成图

逻辑层：BlocMangaer
- 产出seq.block_table
- 职责：维护”每条序列用了哪些 block”

映射层：prepare_prefill / prepare_decode
- 把block_table 映射到slot_mapping: slot = block_table[i] × block_size + offset

物理层：store_kvcache
-  执行写入操作。给定 slot_mapping 和刚算出来的 K/V，在 GPU 上并行地把每个 token 的 K/V 写到 kv_cache tensor 的对应位置

## 全局kv_cache tensor
```
kv_cache shape: (2, 28, 3053, 256, 8, 128)
                 │   │    │     │    │   │
                 │   │    │     │    │   └── head_dim
                 │   │    │     │    └────── num_kv_heads (GQA)
                 │   │    │     └─────────── block_size (每 block 存 256 个 token)
                 │   │    └───────────────── num_blocks (物理 block 总数)
                 │   └────────────────────── num_hidden_layers
                 └────────────────────────── 2: K cache 和 V cache
```

## BlockManager
### allocate
```python
    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            # 取出第i块前缀的token内容
            token_ids = seq.block(i) 
            # 对前缀链做hash
            h = self.compute_hash(token_ids, h) 

            block_id = self.hash_to_block_id[h]

            block = self.blocks[block_id]

            
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            # 分配新的块
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size


```
1. 一段新序列要进入prefill， 这时候只有token_ids, 没有block table
2. can_allocate:
    1. 按 block 切分这条序列的 token，用链式 hash去hash_to_block_id里查，从而找出这条序列与其他序列开头连续有多少个完整 block 能复用
    2. free blocks够不够
3. 把能复用的 block 挂到 seq.block_table 上：正在使用的 block 增加引用计数，空闲但仍保留内容的 block 重新激活；最后再给不能复用的剩余部分分配新 block
> for i in range(seq.num_blocks - 1): 是为了保证使用稳定、可复用的block， 最后一个block可能未满

### append
```python
 if len(seq) % self.block_size == 1:

```
当新生成的 token 成为一个新 block 的第一个 token 时，才需要 append。
### deallocate
采用惰性回收的机制， 只是通过
```python
 self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)
```
来将block设置为可用， 没有在hash_to_block中删除。在后续有相同前缀的序列时可以复用这一块。

## block_table到slot_mapping


