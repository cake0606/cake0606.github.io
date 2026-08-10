# PPO

PPO（Proximal Policy Optimization）是最常见的策略优化方法之一。它解决的并不是“如何让高回报动作更容易被选中”这么单一的问题，而是更具体的一件事：在使用策略梯度更新策略时，如何避免新策略一次偏离旧策略过远，从而让训练保持稳定。

如果只记住 PPO 的 clipped objective，很容易把它理解成一个孤立公式。更自然的理解方式，是把它放回一条完整的训练链里：

`sample action -> collect transition -> estimate value -> build advantage -> freeze old policy -> compute ratio -> clip update`

沿着这条链去看，PPO 里的 actor、critic、advantage、old log-probability 和 clipping 才会真正对应起来。

## PPO 在解决什么问题

普通策略梯度方法会直接沿着“提高高回报动作概率、降低低回报动作概率”的方向更新策略。这个方向本身没有问题，问题在于更新步长缺少显式控制。如果某一次梯度更新过大，新策略就可能迅速偏离旧策略，导致采样分布突然变化，训练随之震荡甚至崩溃。

PPO 的核心思想可以概括为一句话：

> 允许策略变好，但限制它一次不能变得太多。

因此，PPO 真正关心的不是单独某个动作“值不值得鼓励”，而是新策略相对旧策略到底改了多少。这个“改了多少”最终会体现在同一状态、同一动作上的新旧策略概率比里。

## 核心量与关键公式

### 策略函数与价值函数

策略网络表示为：

\[
\pi_\theta(a \mid s)
\]

它输出在状态 \(s\) 下选择动作 \(a\) 的概率分布。

价值网络表示为：

\[
V_\phi(s)
\]

它估计状态 \(s\) 的期望累计回报。PPO 里 critic 的作用不是直接选动作，而是提供价值估计，帮助构造更稳定的训练信号。

### 新旧策略概率比

PPO 的核心量之一是：

\[
r_t(\theta)=\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
\]

这个比值衡量的是：对于已经执行过的动作 \(a_t\)，新策略相对旧策略到底改了多少。

- \(r_t(\theta) > 1\)：新策略更偏向这个动作
- \(r_t(\theta) < 1\)：新策略降低了这个动作的概率

PPO 不会直接禁止这种变化，而是限制这种变化不能太大。

### Advantage

策略更新并不直接依赖原始回报，而是依赖 advantage：

\[
\hat A_t
\]

它描述动作 \(a_t\) 相对于基线的相对好坏。

- \(\hat A_t > 0\)：动作优于基线，应提高概率
- \(\hat A_t < 0\)：动作劣于基线，应降低概率

在最常见的实现里，advantage 往往先从一步 TD 误差出发：

\[
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
\]

再通过 GAE（Generalized Advantage Estimation）递推得到：

\[
\hat A_t = \delta_t + (\gamma \lambda)\delta_{t+1} + (\gamma \lambda)^2\delta_{t+2} + \cdots
\]

这样做的目的，是在偏差和方差之间做折中，让训练信号更稳定。

### Clipped Objective

PPO 最有代表性的目标函数是：

\[
L^{\text{clip}}(\theta)=
\mathbb{E}\left[
\min\left(
r_t(\theta)\hat A_t,\;
\operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat A_t
\right)
\right]
\]

这条公式里最关键的不是 `clip` 这个操作本身，而是它和外层 `min` 一起表达出的约束逻辑：如果更新已经超出安全范围，那么即使继续朝这个方向走，目标函数也不再继续奖励这种过激变化。

这也是 PPO 和普通策略梯度最大的区别。普通策略梯度只关心“应不应该提高这个动作的概率”，PPO 还会额外关心“这次提高得是不是太多了”。

> **补充说明**：论文里常把 PPO 写成“最大化目标”，但代码里通常写成最小化 loss，所以实现中经常出现 `-torch.min(surr1, surr2)`。这个负号只是把最大化目标改写成最小化它的相反数。

## 一条完整的 PPO 执行链

把公式放回训练流程里，PPO 的一个最小更新过程可以写成：

1. 用当前策略和环境交互，得到 `state / action / reward / next_state / done`。
2. 用 critic 计算 \(V(s_t)\) 和 \(V(s_{t+1})\)，构造 `td_target` 与 `td_delta`。
3. 用 `td_delta` 递推得到 advantage。
4. 记录 rollout 时旧策略对已执行动作的 `old_log_probs`，并冻结它。
5. 用当前策略重新计算同一批样本上的 `log_probs`。
6. 用 `exp(log_probs - old_log_probs)` 得到 ratio。
7. 构造 clipped surrogate objective，并和 critic loss 一起优化。

真正把理论和代码串起来的，是下面这条关系：

\[
\hat A_t
\longrightarrow
\log \pi_\theta(a_t \mid s_t),\ \log \pi_{\theta_{\text{old}}}(a_t \mid s_t)
\longrightarrow
r_t(\theta)
\longrightarrow
L^{\text{clip}}(\theta)
\]

只要这条链清楚，PPO 就不会再只是几个零散公式。

## 最小实现

下面给出一份最小但逻辑自洽的离散动作 PPO 实现。它不追求工程完备性，而是尽量让公式、代码和训练流程能一一对上。

```python
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)


class ValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def compute_advantage(gamma: float, lmbda: float, td_delta: torch.Tensor) -> torch.Tensor:
    advantages = []
    advantage = 0.0

    for delta in reversed(td_delta.tolist()):
        advantage = gamma * lmbda * advantage + delta
        advantages.append(advantage)

    advantages.reverse()
    return torch.tensor(advantages, dtype=torch.float32).view(-1, 1)


class PPO:
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        actor_lr: float,
        critic_lr: float,
        gamma: float,
        lmbda: float,
        eps: float,
        epochs: int,
        entropy_coef: float,
        device: str,
    ):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps
        self.epochs = epochs
        self.entropy_coef = entropy_coef
        self.device = device

    def take_action(self, state) -> int:
        state = torch.tensor([state], dtype=torch.float32, device=self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict: Dict[str, List[float]]) -> None:
        states = torch.tensor(transition_dict["states"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(transition_dict["actions"], dtype=torch.long, device=self.device).view(-1, 1)
        rewards = torch.tensor(transition_dict["rewards"], dtype=torch.float32, device=self.device).view(-1, 1)
        next_states = torch.tensor(transition_dict["next_states"], dtype=torch.float32, device=self.device)
        dones = torch.tensor(transition_dict["dones"], dtype=torch.float32, device=self.device).view(-1, 1)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.detach().cpu()).to(self.device)

        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.epochs):
            probs = self.actor(states)
            dist = torch.distributions.Categorical(probs)
            entropy = dist.entropy().view(-1, 1)

            log_probs = torch.log(probs.gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            actor_loss = torch.mean(-torch.min(surr1, surr2) - self.entropy_coef * entropy)
            critic_loss = F.mse_loss(self.critic(states), td_target.detach())

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
```

## 从实现看 PPO 的整体流程

### 先采样，再更新

```python
def take_action(self, state) -> int:
    state = torch.tensor([state], dtype=torch.float32, device=self.device)
    probs = self.actor(state)
    action_dist = torch.distributions.Categorical(probs)
    action = action_dist.sample()
    return action.item()
```

这段代码对应的是：

\[
a_t \sim \pi_\theta(\cdot \mid s_t)
\]

它的作用是从当前策略分布中采样动作，而不是直接选概率最大的动作。训练阶段之所以要保留 `sample()`，是因为策略必须保留随机性，才能维持探索。

这里可以看到，PPO 的 clip 机制并不发生在采样阶段。采样时策略仍然是正常地与环境交互，真正的“限制更新幅度”发生在后面的优化阶段。

### actor 和 critic 分别负责什么

```python
class PolicyNet(nn.Module):
    ...

class ValueNet(nn.Module):
    ...
```

`PolicyNet` 对应 \(\pi_\theta(a \mid s)\)，负责输出动作分布。  
`ValueNet` 对应 \(V_\phi(s)\)，负责给出状态价值估计。

PPO 的训练之所以比最原始的策略梯度更稳定，很大一部分原因就在于这两个角色被拆开了：actor 专门负责“怎么选动作”，critic 专门负责“当前状态值多少钱”，两者在训练中相互配合。

### 从 transition 到 TD 误差

```python
td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
td_delta = td_target - self.critic(states)
```

这里对应的是：

\[
\text{td\_target}_t = r_t + \gamma V_\phi(s_{t+1})(1-d_t)
\]

\[
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
\]

`td_target` 是 critic 要去拟合的目标，而 `td_delta` 则进一步变成构造 advantage 的输入。也就是说，critic 并不是直接参与 clipped objective，而是通过价值估计间接影响 actor 的更新信号。

### GAE 如何把局部 TD 误差变成训练信号

```python
def compute_advantage(gamma: float, lmbda: float, td_delta: torch.Tensor) -> torch.Tensor:
    advantages = []
    advantage = 0.0

    for delta in reversed(td_delta.tolist()):
        advantage = gamma * lmbda * advantage + delta
        advantages.append(advantage)

    advantages.reverse()
    return torch.tensor(advantages, dtype=torch.float32).view(-1, 1)
```

这段代码直接对应 GAE 的递推形式：

\[
\hat A_t = \delta_t + (\gamma \lambda)\delta_{t+1} + (\gamma \lambda)^2\delta_{t+2} + \cdots
\]

倒序循环不是技巧性的写法，而是因为这个定义本身就是从后往前递推的。沿着轨迹从后往前算，可以自然把未来的 TD 误差折回当前时刻。

> **实现细节**：如果直接用完整回报减去 \(V(s_t)\)，理论上也能构造 advantage，但在最常见的 PPO 实现里，`TD + GAE` 这条链通常更稳定。

### 为什么要冻结 `old_log_probs`

```python
old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
```

这里的 `gather(1, actions)` 不是在取整张动作分布，而是在每一个状态对应的那一行里，只取这次真实执行动作 \(a_t\) 的概率。也就是说，这一步拿到的是：

\[
\log \pi_{\theta_{\text{old}}}(a_t \mid s_t)
\]

而 `detach()` 则是 PPO 里非常关键的一个点。因为这里记录的是 rollout 时的旧策略概率，它必须在后续多轮更新中保持不变。只有这样，`ratio` 才真的在表达“新策略相对旧策略改了多少”。

如果不冻结，分母也会跟着当前策略一起动，PPO 的约束意义就不存在了。

### ratio 和 clipped surrogate 如何接上公式

```python
log_probs = torch.log(probs.gather(1, actions))
ratio = torch.exp(log_probs - old_log_probs)

surr1 = ratio * advantage
surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
actor_loss = torch.mean(-torch.min(surr1, surr2) - self.entropy_coef * entropy)
```

这几行就是 PPO 的核心落点。

其中：

\[
r_t(\theta)=
\exp\left(
\log \pi_\theta(a_t \mid s_t) -
\log \pi_{\theta_{\text{old}}}(a_t \mid s_t)
\right)
\]

而 `surr1` 与 `surr2` 则分别对应未截断和截断后的 surrogate objective。到这里为止，前面 actor 采样、critic 估值、GAE 构造 advantage 的所有工作，才真正汇总到一个可优化的目标函数里。

PPO 真正稳定的原因也体现在这里：它不是简单地“按 advantage 改概率”，而是“按 advantage 改概率，同时限制这种改动不能太大”。

### critic loss 和多轮更新的作用

```python
critic_loss = F.mse_loss(self.critic(states), td_target.detach())

for _ in range(self.epochs):
    ...
```

critic loss 对应：

\[
L_V(\phi)=\mathbb{E}\left[(V_\phi(s_t)-\text{td\_target}_t)^2\right]
\]

它的作用是让 critic 更好地逼近状态价值。

而多轮更新则是 PPO 提高样本利用率的重要设计。虽然 PPO 属于 on-policy 方法，但这并不意味着一批样本只能反向传播一次。真正重要的是：这批样本来自旧策略，而且更新时仍然围绕固定的 `old_log_probs` 来做约束。

> **常见误区**：PPO 的 on-policy 性质限制的是“样本必须来自当前 rollout 所对应的旧策略”，而不是“同一批样本绝对不能更新多轮”。

## 实现细节与易混淆点

### 为什么 `gather(1, actions)` 只取一个概率

因为 PPO 的 ratio 只关心“真实执行动作”在新旧策略下的概率变化，而不是整张动作分布都要参与目标函数。

### 为什么 `ratio` 用 `exp(log_probs - old_log_probs)`

因为：

\[
\exp(\log a - \log b)=\frac{a}{b}
\]

实现里保留 log probability 更方便，最后再通过指数恢复成概率比。

### 为什么要加 entropy bonus

只靠 `sample()` 提供的随机性，策略仍然可能很快塌缩成尖锐分布。熵奖励的作用，是延缓这种塌缩，保留更多探索空间。

### 离散动作和连续动作的 PPO 写法不同

本文实现使用 `softmax + Categorical`，只适用于离散动作。连续动作场景通常需要策略网络输出高斯分布参数，而不是离散概率。

## 总结

PPO 的关键不在于背下 clipped objective，而在于能把下面这条链条顺着走通：

`take_action -> td_target -> td_delta -> advantage -> old_log_probs -> ratio -> clipped objective`

一旦这条链条清楚，PPO 里的 actor、critic、GAE、detach、ratio 和 clipping 就不会再是分散的知识点，而会变成一套完整、连贯的训练机制。
