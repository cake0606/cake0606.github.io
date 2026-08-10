# nano-vLLM LLMEngine

## 初始化流程

`LLMEngine.__init__` 依次构建配置、张量并行子进程、主 `ModelRunner`、`self.tokenizer` 和 `Scheduler`：

```python
class LLMEngine:
    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size

        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        self.model_runner = ModelRunner(config, 0, self.events)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id

        self.scheduler = Scheduler(config)
        atexit.register(self.exit)
```

初始化顺序中，`ModelRunner` 必须先于 `Scheduler` 创建。`ModelRunner.__init__` 会执行显存估算，把可分配的 KV Cache 块数写入 `config.num_kvcache_blocks`；随后 `Scheduler` 才能用这个值构造 `BlockManager`。
