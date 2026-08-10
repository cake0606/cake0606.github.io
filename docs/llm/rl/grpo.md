# GRPO

GRPO（Group Relative Policy Optimization）可以看作 PPO 在大语言模型后训练场景中的一种改写。它保留了“限制单次策略更新幅度”的基本思想，但 advantage 的来源不再主要依赖 value model，而是直接来自同一个 query 下多条回答之间的相对 reward。

如果只把 GRPO 记成“PPO + KL”，很容易把它看成若干零散技巧的拼接。更自然的理解方式，是沿着一条完整训练链去看它：

`query -> sample multiple responses -> reward -> group-wise advantage -> score fixed responses -> token logprobs -> ratio / KL -> update policy`

只要这条链条顺下来，GRPO 里的组内标准化、固定 response 打分、token-level ratio、reference KL 和两层 mean 就都会各归其位。

## GRPO 在解决什么问题

语言模型后训练的典型流程，不是“给一个状态选一个动作”，而是：

1. 给定一个 query。
2. 当前策略对这个 query 采样多条 response。
3. 用规则、verifier 或 reward model 对这些 response 打分。
4. 根据打分结果更新策略。

如果只按 reward 高低直接提高或降低 response 的概率，策略更新仍然可能过于剧烈。模型会很快偏离 rollout 时的分布，训练稳定性也会随之下降。GRPO 与 PPO 一样，首先要解决的就是这个问题：既要让更好的回答获得更高概率，又要限制单次更新幅度。

与 PPO 常见实现不同的是，GRPO 通常不依赖 critic 去估计 baseline，而是直接在同一个 query 对应的一组回答内部做相对比较。某条 response 的“好坏”因此不只是绝对 reward 的问题，而是它相对同组其他 response 的位置。

## 核心量与关键公式

### 一组回答与回答级别 reward

对同一个 query \(x\)，当前策略会采样出一组回答：

\[
\{y_1, y_2, \dots, y_G\}
\]

其中 \(G\) 对应实现里的 `group_size`。

每条回答 \(y_i\) 对应一个回答级别的 reward：

\[
r_i
\]

这个 reward 可以来自规则、reward model 或 verifier。它的作用是评价整条 response 的质量，而不是直接参与 token 级别的概率比计算。

### 组内 advantage

GRPO 的关键变化之一，是 advantage 不再由 value function 估计，而是直接在组内对 reward 做标准化。最常见的形式是：

\[
A_i = \frac{r_i - \mu_{\text{group}}}{\sigma_{\text{group}} + \epsilon}
\]

其中：

- \(\mu_{\text{group}}\) 是当前组 reward 的均值
- \(\sigma_{\text{group}}\) 是当前组 reward 的标准差
- \(\epsilon\) 是防止除零的稳定项

这条公式表达的其实很直接：一条回答值不值得在本轮更新中被鼓励，不是单独看它的绝对 reward，而是看它相对组内平均水平高多少或低多少。

> **实现细节**：这里是减均值后除以标准差，不是除以方差。  
> **常见误区**：标准化范围是组内，而不是整个 batch。如果把不同 query 的回答混在一起归一化，就不再是在比较“同一题目下谁更好”。

### token-level log probability

虽然 reward 和 advantage 都是回答级别的量，但策略更新仍然发生在 token 级别。对一条固定回答 \(y=(y_1,\dots,y_T)\)，模型在第 \(t\) 个 token 上的 log probability 表示为：

\[
\log \pi_\theta(y_t \mid x, y_{<t})
\]

这意味着 GRPO 真正优化的，不是一个抽象的“整条 response 概率”，而是这条 response 里每个生成 token 在当前策略下的条件概率。

### 新旧策略概率比

和 PPO 一样，GRPO 的核心更新量仍然是新旧策略概率比：

\[
r_t(\theta)=
\frac{\pi_\theta(y_t \mid x, y_{<t})}
{\pi_{\theta_{\text{old}}}(y_t \mid x, y_{<t})}
\]

在实现中，它通常通过 log probability 的差值来计算：

\[
r_t(\theta)=
\exp\left(
\log \pi_\theta(y_t \mid x, y_{<t})
-
\log \pi_{\theta_{\text{old}}}(y_t \mid x, y_{<t})
\right)
\]

这就是为什么代码里会先拿到 `new_lp` 和 `old_lp`，再做 `torch.exp(new_lp - old_lp)`。

### reference KL

为了防止当前策略偏离参考分布太远，GRPO 常常还会引入 reference model，并加入一个 KL 惩罚项。一个常见的 token-level 估计器是：

\[
\mathrm{KL}_t \approx
\exp(\log \pi_{\text{ref}} - \log \pi_\theta)
- (\log \pi_{\text{ref}} - \log \pi_\theta) - 1
\]

对应到实现里就是：

```python
delta = ref_lp - new_lp
kl = torch.exp(delta) - delta - 1
```

这一步的作用不是增强探索，而是限制当前策略偏离参考模型太远。

> **补充说明**：reference KL 和 PPO 中常见的 entropy bonus 不是一回事。entropy bonus 主要用来维持分布宽度，reference KL 则更像是在限制策略漂移。

### GRPO 的目标函数

把这些量放在一起，一个常见的 GRPO-style objective 可以写成：

\[
\mathcal{L}_{\text{GRPO}} =
\mathbb{E}\left[
\min\left(r_t A_i,\ \operatorname{clip}(r_t, 1-\epsilon, 1+\epsilon)A_i\right)
- \beta \,\mathrm{KL}_t
\right]
\]

其中：

- \(r_t\) 是 token 级别的新旧策略概率比
- \(A_i\) 是回答 \(i\) 的组内 advantage
- \(\epsilon\) 是 clipping 范围
- \(\beta\) 是 KL 惩罚系数

这里最容易让人卡住的地方，是回答级别量和 token 级别量如何结合。实际上，GRPO 在最小实现里通常采用一种很直接的处理：同一条回答内部的所有生成 token，共享这条回答对应的 advantage。

## 一条完整的 GRPO 执行链

如果把公式放回训练流程，GRPO 的一个最小更新链条可以写成：

1. 对同一个 query 采样一组 response。
2. 为每条 response 计算 reward。
3. 在组内把 reward 标准化成 advantage。
4. 固定这些 response，不再重新生成。
5. 让当前策略、旧策略和 reference model 对同一批固定 response 逐 token 打分。
6. 用 `ratio + clipping + reference KL` 构造 loss。
7. 先在 token 级别聚合，再在回答级别和 batch 级别聚合，最后更新当前策略。

这条链里最关键的前提只有一个：`new_logprobs`、`old_logprobs` 和 `ref_logprobs` 必须都来自同一条固定 response。否则一旦不同模型生成的 token 序列发生错位，ratio 和 KL 就不再有意义。

## 最小实现

下面给出一份最小但完整的 GRPO 示例实现。它聚焦于“给定已采样好的 trajectories，如何完成一次 GRPO 更新”，而不展开完整的 generation 系统。

```python
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn


@dataclass
class TokenStep:
    token_id: int
    token_text: str
    log_prob: float
    position: int


@dataclass
class Trajectory:
    query: str
    token_steps: List[TokenStep]
    generated_text: str
    reward: float
    final_answer: str
    full_input_ids: List[int]
    generated_positions: List[int]


class GRPOLoss(nn.Module):
    def __init__(
        self,
        group_size: int = 4,
        clip_eps: float = 0.2,
        kl_coef: float = 0.01,
        adv_eps: float = 1e-8,
    ):
        super().__init__()
        self.group_size = group_size
        self.clip_eps = clip_eps
        self.kl_coef = kl_coef
        self.adv_eps = adv_eps

    def compute_kl_divergence(self, new_lp: torch.Tensor, ref_lp: torch.Tensor) -> torch.Tensor:
        delta = ref_lp - new_lp
        return torch.exp(delta) - delta - 1

    def forward(
        self,
        new_logprobs: List[torch.Tensor],
        old_logprobs: List[torch.Tensor],
        advantages: torch.Tensor,
        ref_logprobs: List[torch.Tensor],
    ) -> torch.Tensor:
        if len(new_logprobs) != len(old_logprobs) or len(new_logprobs) != len(ref_logprobs):
            raise ValueError("Batch sizes of new_logprobs, old_logprobs, and ref_logprobs must match.")
        if len(advantages) != len(new_logprobs):
            raise ValueError("advantages must provide one scalar per response.")

        all_loss = []

        for i in range(len(new_logprobs)):
            new_lp = new_logprobs[i].reshape(-1)
            old_lp = old_logprobs[i].reshape(-1)
            ref_lp = ref_logprobs[i].reshape(-1)
            adv = advantages[i]

            if len(new_lp) != len(old_lp) or len(new_lp) != len(ref_lp):
                raise ValueError(
                    f"Token lengths must match for sample {i}, "
                    f"got new={len(new_lp)}, old={len(old_lp)}, ref={len(ref_lp)}."
                )

            ratio = torch.exp(new_lp - old_lp)
            token_adv = adv.expand_as(ratio)

            surr1 = ratio * token_adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * token_adv

            kl_loss = self.compute_kl_divergence(new_lp, ref_lp)
            token_loss = -torch.min(surr1, surr2) + self.kl_coef * kl_loss

            all_loss.append(token_loss.mean())

        return torch.stack(all_loss).mean()


class GRPOTrainer:
    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        optimizer,
        loss_fn: GRPOLoss,
        group_size: int = 4,
        device: str = "cpu",
        adv_eps: float = 1e-8,
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.group_size = group_size
        self.device = device
        self.adv_eps = adv_eps

    def compute_advantages(self, rewards: List[float]) -> torch.Tensor:
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        if len(rewards_tensor) == 0:
            raise ValueError("rewards cannot be empty.")
        if len(rewards_tensor) % self.group_size != 0:
            raise ValueError(
                f"Number of rewards ({len(rewards_tensor)}) must be divisible by group_size ({self.group_size})."
            )

        grouped_rewards = rewards_tensor.view(-1, self.group_size)
        group_mean = grouped_rewards.mean(dim=1, keepdim=True)
        group_std = grouped_rewards.std(dim=1, unbiased=False, keepdim=True)

        grouped_advantages = (grouped_rewards - group_mean) / (group_std + self.adv_eps)
        return grouped_advantages.reshape(-1)

    def compute_logprobs(self, model: nn.Module, trajectories: List[Trajectory]) -> List[torch.Tensor]:
        all_logprobs = []

        for traj in trajectories:
            input_ids = torch.tensor(
                traj.full_input_ids,
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)

            outputs = model(input_ids)
            logits = outputs.logits.squeeze(0)
            log_probs = torch.log_softmax(logits, dim=-1)

            token_logprobs = []
            for pos in traj.generated_positions:
                if pos <= 0:
                    raise ValueError(f"Generated token position must be > 0, got {pos}.")
                if pos >= input_ids.shape[1]:
                    raise ValueError(
                        f"Generated token position {pos} is out of range for input length {input_ids.shape[1]}."
                    )

                token_id = input_ids[0, pos]
                token_lp = log_probs[pos - 1, token_id]
                token_logprobs.append(token_lp)

            all_logprobs.append(torch.stack(token_logprobs))

        return all_logprobs

    def update_step(
        self,
        trajectories: List[Trajectory],
        old_logprobs: List[torch.Tensor],
    ) -> torch.Tensor:
        if len(trajectories) == 0:
            raise ValueError("trajectories cannot be empty.")
        if len(old_logprobs) != len(trajectories):
            raise ValueError(
                f"old_logprobs batch size ({len(old_logprobs)}) must match trajectories batch size ({len(trajectories)})."
            )

        self.policy_model.train()
        self.ref_model.eval()

        rewards = [traj.reward for traj in trajectories]
        advantages = self.compute_advantages(rewards)

        new_logprobs = self.compute_logprobs(self.policy_model, trajectories)
        with torch.no_grad():
            ref_logprobs = self.compute_logprobs(self.ref_model, trajectories)

        loss = self.loss_fn(new_logprobs, old_logprobs, advantages, ref_logprobs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
```

## 从实现看 GRPO 的整体流程

### `Trajectory` 在保存什么

```python
@dataclass
class Trajectory:
    query: str
    token_steps: List[TokenStep]
    generated_text: str
    reward: float
    final_answer: str
    full_input_ids: List[int]
    generated_positions: List[int]
```

`Trajectory` 表示一条固定的 query-response 样本，以及这条样本在训练时需要用到的信息。

这里最关键的字段其实只有三个：

- `reward`：整条回答的评分
- `full_input_ids`：完整 token 序列，用于重新打分
- `generated_positions`：哪些位置属于生成 token

它们共同决定了后面能否对这条固定 response 逐 token 计算 log probability。

> **实现细节**：`generated_positions` 必须是 `full_input_ids` 里的绝对位置，而不是生成片段内部的相对位置。否则 `pos - 1` 的对齐关系会被破坏。

### reward 如何变成 advantage

```python
def compute_advantages(self, rewards: List[float]) -> torch.Tensor:
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
    grouped_rewards = rewards_tensor.view(-1, self.group_size)
    group_mean = grouped_rewards.mean(dim=1, keepdim=True)
    group_std = grouped_rewards.std(dim=1, unbiased=False, keepdim=True)
    grouped_advantages = (grouped_rewards - group_mean) / (group_std + self.adv_eps)
    return grouped_advantages.reshape(-1)
```

这段代码直接对应：

\[
A_i = \frac{r_i - \mu_{\text{group}}}{\sigma_{\text{group}} + \epsilon}
\]

它的位置也非常明确：先从回答级别 reward 出发，再把它转成回答级别 advantage。GRPO 这里并不需要 critic，因为训练信号已经来自组内相对差异。

### 为什么模型可以对固定 response 逐 token 打分

```python
outputs = model(input_ids)
logits = outputs.logits.squeeze(0)
log_probs = torch.log_softmax(logits, dim=-1)

for pos in traj.generated_positions:
    token_id = input_ids[0, pos]
    token_lp = log_probs[pos - 1, token_id]
```

语言模型本质上建模的是：

\[
P(y_t \mid x, y_{<t})
\]

因此，只要给定完整前缀，模型就能在每个位置输出“下一个 token 的整张词表分布”。生成时是从这个分布里采样一个 token，而这里做的是另外一件事：直接读取真实 token 在该分布下的 log probability。

这也解释了为什么不能只取 `log_probs[pos - 1]`。因为它是一整张词表分布，真正需要的是其中真实 token 对应的那一个值，所以还必须再用 `token_id` 索引一次。

> **补充说明**：这里之所以是 `pos - 1`，是因为自回归模型在位置 \(t\) 的输出，用来预测位置 \(t+1\) 的 token。  
> **常见误区**：`pos == 0` 在这里本身就是不合理的，因为序列第一个 token 没有可用的前缀去预测它。

### 回答级别 advantage 如何进入 token-level loss

```python
ratio = torch.exp(new_lp - old_lp)
token_adv = adv.expand_as(ratio)

surr1 = ratio * token_adv
surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * token_adv

kl_loss = self.compute_kl_divergence(new_lp, ref_lp)
token_loss = -torch.min(surr1, surr2) + self.kl_coef * kl_loss
```

这里对应的就是：

\[
\mathcal{L}_{\text{GRPO}} =
\mathbb{E}\left[
\min\left(r_t A_i,\ \operatorname{clip}(r_t, 1-\epsilon, 1+\epsilon)A_i\right)
- \beta \,\mathrm{KL}_t
\right]
\]

`adv` 是一条回答对应的标量，`ratio` 是这条回答中每个生成 token 的向量。`adv.expand_as(ratio)` 的意思并不是“每个 token 有自己的 reward”，而是“这条回答整体被评价为更好或更差，因此它内部所有生成 token 在这轮更新中共享同一个优势方向”。

### 为什么会有两层 `mean`

```python
all_loss.append(token_loss.mean())
return torch.stack(all_loss).mean()
```

这两层聚合分别对应两件事：

1. 先对单条回答内部所有生成 token 求平均
2. 再对一个 batch 内的多条回答求平均

也就是说，第一层是从 token-level 到 answer-level，第二层才是从 answer-level 到 batch-level。

### `update_step` 如何把整条链串起来

```python
rewards = [traj.reward for traj in trajectories]
advantages = self.compute_advantages(rewards)

new_logprobs = self.compute_logprobs(self.policy_model, trajectories)
with torch.no_grad():
    ref_logprobs = self.compute_logprobs(self.ref_model, trajectories)

loss = self.loss_fn(new_logprobs, old_logprobs, advantages, ref_logprobs)
```

这一段几乎就是前面理论主线在代码里的压缩版：

1. 提取 reward
2. 计算组内 advantage
3. 当前策略对固定 response 打分
4. reference model 对同一批 response 打分
5. 连同 rollout 时保存的 `old_logprobs` 一起进入 loss
6. 反向传播并更新当前策略

`old_logprobs` 没有在这里重新计算，是因为它代表 rollout 时的旧策略分布，它应该被当作一个冻结的比较基准。

## 实现细节与易混淆点

### 为什么 old / new / ref 必须评分同一条 response

因为 ratio 和 KL 的前提是：比较的对象必须是同一个条件分布下的同一个 token。如果让不同模型各自重新生成 response，只要其中一个模型多一个 token 或少一个 token，后面的序列就会整体错位，逐 token 比较立即失去意义。

### `ref_model` 为什么要单独保留

即使 reference model 初始参数来自旧模型或当前模型的拷贝，它在训练逻辑上也应是一个单独对象。因为它承担的是“稳定参考分布”的角色，而不是参与当前这轮参数更新。

## 总结

GRPO 的重点并不在于引入了一个完全不同的优化器，而是在 PPO 风格的受限更新框架上，把训练信号从“critic 估计 advantage”转成了“组内 reward 相对差异”，并进一步用 reference KL 约束策略漂移。

真正把 GRPO 理清楚，关键是顺着这条链条走：

`sample responses -> reward -> group-wise advantage -> score fixed responses -> ratio / KL -> token loss -> answer mean -> batch mean`

当这条链条顺下来之后，为什么要组内标准化、为什么可以对固定 response 计算 logprobs、为什么要用 `log_probs[pos - 1, token_id]`、为什么会有两层 `mean`，这些问题都会自然落到各自的位置上。
