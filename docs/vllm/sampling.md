## greedy Sampling
### 参数定义
在用户构造一个SamplingParams(temperature=0, n=1)时：
1.  进行参数检查， 如果温度<0就返回false
2. 通过两个阈值常量来划分温度区间：
_SAMPLING_EPS = 1e-5
_MAX_TEMP = 1e-2

通过温度区间划分GREEDY和RANDOM：
```py
    @cached_property
    def sampling_type(self) -> SamplingType:
        if self.temperature < _SAMPLING_EPS:
            return SamplingType.GREEDY
        if self.seed is not None:
            return SamplingType.RANDOM_SEED
        return SamplingType.RANDOM
```
  - temp < 1e-5 → GREEDY
  - 1e-5 ≤ temp < 0.01 → 钳位到 0.01，仍是 RANDOM
  - temp ≥ 0.01 → 正常 RANDOM
使用1e-5而不是==0来判断GREEDY， 是为了浮点安全， 如果用户传1e-6这种极小值， 在softmax中无意义， 直接判定为greedy

使用SamplingType来枚举
```py
class SamplingType(IntEnum):
    GREEDY = 0
    RANDOM = 1
    RANDOM_SEED = 2
```
GREEDYF放在第一位， 处于最优先判定。

3. 在判定为GREEDY时， 对参数进行强制规整
```py
        if self.temperature < _SAMPLING_EPS:
            # Zero temperature means greedy sampling.
            self.top_p = 1.0
            self.top_k = 0
            self.min_p = 0.0
            self._verify_greedy_sampling()
```
其中
  - top_p = 1.0（保留全部概率质量，不截断）
  - top_k = 0（vLLM 约定 0 = 不截断）
  - min_p = 0.0（不设下限)

4. 后续代码在访问params.sampling_type -> cached_property返回GREEDY, 且参数是规整后的干净状态

## Batch聚合
在一个batch中同时混合了greed和random请求时， 在不拆batch的前提下， 使greedy走最省算力的路径。
1. 在add_request中注册：
```py
        if sampling_params := request.sampling_params:
            if sampling_params.sampling_type == SamplingType.GREEDY:
                # Should avoid division by zero later when apply_temperature.
                self.temperature_cpu[req_index] = 0.0
                self.greedy_reqs.add(req_id)
            else:
                self.temperature_cpu[req_index] = sampling_params.temperature
                self.random_reqs.add(req_id)
```
每个请求加入batch时， 会按照sampling_type加入到对应的集合中。
在GREEDY分支中显式将temperature_cpu[req_index] = 0， 是为了下游使用torch.where时有确定的0 。

2. 节省一次temperature张量拷贝
```py
    def _make_sampling_metadata(self) -> SamplingMetadata:
        num_reqs = self.num_reqs
        if not self.all_greedy:
            temperature = copy_slice(
                self.temperature_cpu_tensor, self.temperature, num_reqs
            )
        else:
            temperature = None
```
在all greedy的时候， 直接将temperature设置为None， 不传这个张量。
在sample/sampler.py中
```py
  if sampling_metadata.all_random:
            greedy_sampled = None
        else:
            greedy_sampled = self.greedy_sample(logits)
            if sampling_metadata.all_greedy:
                processed_logprobs = None
                if (
                    sampling_metadata.max_num_logprobs is not None
                    or sampling_metadata.logprob_token_ids
                ):
                    if logprobs_mode == "processed_logits":
                        processed_logprobs = logits
                    elif logprobs_mode == "processed_logprobs":
                        processed_logprobs = self.compute_logprobs(logits)
                return greedy_sampled, processed_logprobs

        assert sampling_metadata.temperature is not None
```
这样保证了：
 temperature is None ⟺ all_greedy=True ⟺ sample() 会在 assert 之前 return。

3. 混合batch的处理
首先对logits进行argmax：
```py
 greedy_sampled = self.greedy_sample(logits)
```
```py
    @staticmethod
    def apply_temperature(
        logits: torch.Tensor,
        temp: torch.Tensor,
        all_random: bool,
    ) -> torch.Tensor:
        # Use in-place division to avoid creating a new tensor.
        # Avoid division by zero if there are greedy requests.
        if not all_random:
            temp = torch.where(temp < _SAMPLING_EPS, 1.0, temp)
        return logits.div_(temp.unsqueeze(dim=1))
``` 
apply_temperature会先对温度小于_SAMPLING_EPS的temp设置为1.0， 防止出现div 0， 然后对整个logits进行div temp的操作。

通过processor对logits进行处理：
```py
        # Apply logits processors that only apply to random sampling
        # (argmax invariant)
        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            logits = processor.apply(logits)
```
> 生成token的整体流程是： model forward -> logits(一个词表) -> processor加工logits -> 采样 -> 选出一个token
processor对原始分数logits做某种修改， 例如：
  - MinPLogitsProcessor：按概率屏蔽低分 token
  - MinTokensLogitsProcessor：屏蔽 EOS（强制至少生成 N 个 token 再停）
  - LogitBiasLogitsProcessor：给指定 token 加减分数
argmax_invariant指在processor应用之后， 分数最大的token不变。
对于greedy来说， min_p对它无效。对random来说， 保留概率>= max_prob * min_p的token。

对logits进行topk-topp处理：
```py
 logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
```
首先按照logits分数进行升序排序， 用logits_idx维护分数与token_id的对应关系。
```py
 if k is not None:
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)  # shape: B
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))
```
top k保留前k大：
    1. 计算第k名的索引
    2. gather出第k名的值作为阈值
    3. 比阈值小的设置为-inf
```py
 if p is not None:
        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))
```
top p按累积概率保留：
   1. 对排序后的logits进行一次softmax
   2. 计算升序累积和
   3. 累积值<= 1-p 的屏蔽
   4. 末尾一定保留

```py
    # Re-sort the probabilities.
    return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)
```
把logits_sort的值写回logits。

```py
        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,  # Reuse tensor
        )
```

