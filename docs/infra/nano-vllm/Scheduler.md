---
id: Scheduler
aliases: []
tags: []
---

## Sequence解析
Sequence是一条请求的状态机。
### 字段
seq_id	全局唯一 ID，自增分配
token_ids	当前完整序列（prompt + 已生成的 token）
last_token	最后一个 token（decode 时只需要算这一个的 Q）
num_tokens	当前总长度
num_prompt_tokens	prompt 部分的长度（不变）
num_cached_tokens	前缀中命中 prefix cache 的 token 数（本篇不展开）
block_table	该序列占用的物理 KV 块 ID 列表
status	WAITING / RUNNING / FINISHED
temperature / max_tokens / ignore_eos	从 SamplingParams 拷贝过来的采样参数

## BlockManager
提供接口
方法	问题	谁调用
can_allocate(seq)	空闲块够不够给这条 seq 当前长度分配？	prefill 准入判断
allocate(seq)	给 seq 填 block_table，可能更新 num_cached_tokens	从 waiting 拉出来时
can_append(seq)	下一步 decode 会不会跨块、需要新块？够不够？	decode 准入判断
may_append(seq)	如果跨块了，分配新块；如果当前块刚满，固化 hash	decode 分支里拉入前
deallocate(seq)	释放 seq 占用的所有块（ref_count–，归零则回收）	序列结束或被 preempt 时
## schedule
### prefill 阶段
计算需要prefill多少token
- 这个请求还没有正式分配过 block, 通过can_allocate(seq)估计：

    - 这个 prompt 的前面有多少个完整 block 可以直接复用 prefix cache
    - 这些可复用 block 对应多少已经“等价于缓存好的” token
    计算出的总 token 数 - 可复用的 prefix cache token 数
- 这个请求已经有 block_table 了，说明它不是第一次调度：
    - 之前做过一部分 chunked prefill，还没做完
    - 或者已经分配过 block，当前正在继续补剩余 prompt
    总 token 数 - 当前已经实际 cached 的 token 数
```python
if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
```
只允许一个请求被切块，不允许后续请求也以半截形式进入

### decode阶段
1. 从队首取出一个running请求
2. 判断能不能append一个token
 - kv cache还有空间
    ```python
    seq.num_scheduled_tokens = 1
seq.is_prefill = False
self.block_manager.may_append(seq)
scheduled_seqs.append(seq)
    ```
    成功调度并且生成一个token
    - 没有空间时
        - 不断取出别的活跃请求， 抢占他们的block：
            取出这个请求， 执行preemt
        - 没有请求就preempt自己
3. 把处理的seq调整顺序后放入running 队列

## postprocess
对每个执行过的 seq:
    把新形成的完整 block 注册进 prefix cache
    更新这个 seq 已缓存的 token 数
    如果这轮还是未完成的 prefill:
        只更新进度，不产生输出 token
    否则:
        把 sampled token 真正接到序列后面
        如果遇到 eos 或达到 max_tokens:
            标记 finished
            释放 KV cache
            从 running 移除

