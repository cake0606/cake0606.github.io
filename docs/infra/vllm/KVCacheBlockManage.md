# 数据结构定义
## KVCacheBlock
在vllm中， 使用`KVCacheBlock`来存储metadata
```python
@dataclass(slots=True)
class KVCacheBlock:
    """KV-cache block metadata."""

    block_id: int

    ref_cnt: int = 0

    # 在block为full并且cached时， 记录hash key
    _block_hash: BlockHashWithGroupId | None = None

    # 被_block_hash所覆盖的prefix token数量
    _block_hash_num_tokens: int | None = None

    # FreeKVCacheBlockQueue通过这两个指针来操控双向链表
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None

    is_null: bool = False
```
## FreeKVCacheBlockQueue
vllm让KVCacheBlock本身带有：
- prev_free_block
- next_free_block
而不是使用deque， 原因是deque.remove(block)是o(n)的操作。
通过使用fake head 和fake tail来减少对双向链表操作时产生的分支
FreeKVCacheBlockQueue的职责为：
- 空闲物理block集合
- prefix cache的LRU淘汰队列

### 空闲block重新入队规则
1. 在启用`prefix cache`时， 将无hash block加入到队首, 原因是无hash block:
    - 不能产生prefix cache hit 
    - 优先被下一次分配覆盖
2. 有hash block加到队尾
    - 未来请求仍可能命中， 放在队尾延迟覆盖
3. 关闭`prefix cache`时， 统一加入队尾, 这时`FreeKVCacheBlockQueue`不在承担prefix LRU， 只是作为普通循环队列。
    ```python
    if block.block_hash is None and self.enable_caching:
      blocks_without_hash.append(block)
    else:
       blocks_with_hash.append(block)
    ```




