---
id: llm_engine
aliases: []
tags: []
---

## init流程
```python
class LLMEngine:
    def __init__(self, model, **kwargs):
        # 1. 构建 Config
        config = Config(model, **config_kwargs)

        # 2. 若 TP > 1，spawn 子进程（本篇不展开）
        for i in range(1, config.tensor_parallel_size):
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()

        # 3. 创建 ModelRunner（rank 0）—— 加载模型、warmup、分配 KV cache、录 CUDA Graph
        self.model_runner = ModelRunner(config, 0, self.events)

        # 4. 加载 tokenizer，拿到 eos_token_id
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        config.eos = self.tokenizer.eos_token_id

        # 5. 创建 Scheduler —— 注意：必须在 ModelRunner 之后
        self.scheduler = Scheduler(config)

        # 6. 注册退出清理
        atexit.register(self.exit)
```
> ModelRunner在init时会通过显存估算计算出能分配的kvcache_block数量
 
