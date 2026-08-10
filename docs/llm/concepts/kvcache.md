# KV Cache

KV Cache 可以理解为大模型在自回归推理阶段保存的“历史注意力记忆”。如果不做 cache，那么每生成一个新 token，都要把整段历史重新过一遍模型；做了 cache 之后，历史 token 在各层的 \(K/V\) 可以直接复用，新 step 只需要为当前 token 计算新的 \(q/k/v\)，再把新的 \(K/V\) 追加进去。

`prompt -> prefill -> 写入各层 KV cache -> 取最后位置 logits -> 采样 next token -> decode -> 复用旧 KV 并追加新 KV`

## 1. KV Cache 在解决什么问题

自回归生成的目标是：

\[
P(x_{T+1} \mid x_1, x_2, \dots, x_T)
\]

也就是给定当前上下文，预测下一个 token。问题在于，Transformer 的每一层 attention 都需要读取当前 token 之前的历史信息。如果不做缓存，那么每生成一个新 token，都要把历史 token 在每一层重新计算一次。

以长度为 \(T\) 的上下文为例，不做 cache 时：

1. 为了得到 \(x_{T+1}\)，要重新计算 \(x_1,\dots,x_T\) 的整段前向。
2. 为了得到 \(x_{T+2}\)，又要重新计算 \(x_1,\dots,x_T,x_{T+1}\) 的整段前向。
3. 后续每一步都继续重复这件事。

真正重复计算的核心不是“历史 token 的最终 logits”，而是它们在每一层已经算过的注意力中间量。对后续 token 来说，历史 token 最有价值的中间结果就是各层的 \(K/V\)，因为新 token 做 attention 时会持续读取这些历史 \(K/V\)。

因此 KV Cache 的核心作用就是：

\[
\boxed{
\text{把历史 token 在各层的 } K/V \text{ 保存下来，后续 step 直接复用}
}
\]

这样一来，历史 token 不需要在每步生成时反复重算，推理效率会明显提高。

## 2. 核心量与关键公式

### 2.1 单层 attention 在计算什么

先只看第 \(l\) 层、第 \(i\) 个位置。对单个 attention head 而言：

\[
q_i^{(l)} \in \mathbb{R}^{d_h}, \qquad
k_j^{(l)} \in \mathbb{R}^{d_h}, \qquad
v_j^{(l)} \in \mathbb{R}^{d_h}
\]

其中 \(d_h\) 是单个 head 的维度。

第 \(i\) 个位置的 attention 输出可以写成：

\[
o_i^{(l)} = \sum_{j \le i} \alpha_{ij}^{(l)} v_j^{(l)},
\qquad
o_i^{(l)} \in \mathbb{R}^{d_h}
\]

其中注意力权重为：

\[
\alpha_{ij}^{(l)} =
\frac{
\exp\left(\frac{q_i^{(l)} {k_j^{(l)}}^\top}{\sqrt d_h}\right)
}{
\sum_{m \le i}\exp\left(\frac{q_i^{(l)} {k_m^{(l)}}^\top}{\sqrt d_h}\right)
}
\]

这里有三个直接结论。

第一，当前位置 \(i\) 在本层只直接使用自己的 \(q_i^{(l)}\)。

第二，当前位置会读取所有可见历史位置的 \(K/V\)，也就是：

\[
\{k_1^{(l)}, \dots, k_i^{(l)}\}, \qquad
\{v_1^{(l)}, \dots, v_i^{(l)}\}
\]

第三，同层前面位置的 \(q_j^{(l)}\) 不直接进入当前位置的 attention 公式。后面 token 真正需要反复读取的是历史位置的 \(K/V\)，这正是 KV Cache 缓存对象的来源。

如果把一个 head 扩展到整段序列，常见张量形状可以写成：

\[
Q^{(l)}, K^{(l)}, V^{(l)} \in \mathbb{R}^{T \times d_h}
\]

如果带 batch 和多头，一个常见实现形状是：

\[
Q, K, V \in \mathbb{R}^{B \times H \times T \times d_h}
\]

其中：

- \(B\)：batch size
- \(H\)：attention 头数
- \(T\)：序列长度

### 2.2 一层里的 hidden state 是怎么更新的

设第 \(l-1\) 层的输入 hidden states 为：

\[
H^{(l-1)} \in \mathbb{R}^{T \times d_{\text{model}}}
\]

在 decoder-only Transformer 里，一层通常由 attention 和 MLP 两部分组成。用 pre-norm 形式写，简化后的单层更新可以写成：

\[
U^{(l)} = H^{(l-1)} + \operatorname{Attention}(\operatorname{Norm}(H^{(l-1)})),
\qquad
U^{(l)} \in \mathbb{R}^{T \times d_{\text{model}}}
\]

\[
H^{(l)} = U^{(l)} + \operatorname{MLP}(\operatorname{Norm}(U^{(l)})),
\qquad
H^{(l)} \in \mathbb{R}^{T \times d_{\text{model}}}
\]

这里需要区分两个量。

attention 子层输出是：

\[
O^{(l)} = \operatorname{Attention}(\operatorname{Norm}(H^{(l-1)})),
\qquad
O^{(l)} \in \mathbb{R}^{T \times d_{\text{model}}}
\]

而真正这一层结束后的 hidden state 是：

\[
H^{(l)}
\]

也就是说，hidden state 不是单纯的“softmax 注意力分数乘 value”，而是 attention、残差、归一化、MLP 共同作用之后的结果。最后一层最后一个位置的 hidden state 再经过输出投影，才会变成 next-token logits。

### 2.3 为什么只缓存 K/V，不缓存 Q

对新生成 token 来说，在第 \(l\) 层真正需要的 attention 计算是：

\[
o_{T+1}^{(l)}
=
\operatorname{softmax}
\left(
\frac{
q_{T+1}^{(l)} {K_{\le T+1}^{(l)}}^\top
}{\sqrt d_h}
\right)
V_{\le T+1}^{(l)}
\]

这里：

- \(q_{T+1}^{(l)} \in \mathbb{R}^{1 \times d_h}\)
- \(K_{\le T+1}^{(l)} \in \mathbb{R}^{(T+1) \times d_h}\)
- \(V_{\le T+1}^{(l)} \in \mathbb{R}^{(T+1) \times d_h}\)
- 输出 \(o_{T+1}^{(l)} \in \mathbb{R}^{1 \times d_h}\)

当前 step 只需要当前 token 自己的 \(q_{T+1}^{(l)}\)，而历史部分真正需要被持续读取的是 \(K_{\le T}^{(l)}\) 和 \(V_{\le T}^{(l)}\)。

历史 token 的 \(q\) 在后续 step 里不会再被拿出来参与新的 attention。它只在该 token 当时被计算 hidden state 的那一刻有用，之后就不再是后续 token 必需的中间量。

所以从复用价值看：

\[
\boxed{
\text{历史 } Q \text{ 是一次性量，历史 } K/V \text{ 是后续 step 持续复用的量}
}
\]

因此推理缓存只需要保存 \(K/V\)，而不需要保存历史 \(Q\)。

### 2.4 KV Cache 的显存占用

KV Cache 的显存主要由以下几个因素决定：

- batch size
- 当前缓存的 token 数
- 层数
- KV 头数
- 每个 head 的维度
- 数据类型字节数

常见估算公式是：

\[
\text{KV bytes}
=
B \times T \times L \times H_{kv} \times D \times 2 \times s
\]

其中：

- \(B\)：batch size
- \(T\)：当前 cache 中已保存的 token 数
- \(L\)：Transformer 层数
- \(H_{kv}\)：KV 头数
- \(D\)：每个 KV head 的维度
- \(2\)：分别对应 \(K\) 和 \(V\)
- \(s\)：每个元素占用的字节数，FP16/BF16 通常为 2

这个公式对应的单层单样本 KV 张量形状，可以理解为：

\[
K^{(l)}, V^{(l)} \in \mathbb{R}^{H_{kv} \times T \times D}
\]

如果把 batch 和层数都算进去，就是：

\[
\text{all KV} \sim [L, B, 2, H_{kv}, T, D]
\]

这里只是帮助理解维度来源，实际工程实现不一定按这个顺序存放。

这个公式有几个重要特征。

第一，KV Cache 对上下文长度 \(T\) 是线性增长的。

第二，KV Cache 对 batch size \(B\) 也是线性增长的。

第三，是否采用 MHA、GQA、MQA，会直接影响 \(H_{kv}\)，从而显著影响显存。

如果是普通 MHA，通常有：

\[
H_{kv} = H_q
\]

如果是 GQA 或 MQA，则：

\[
H_{kv} < H_q
\]

这也是很多大模型在长上下文推理时使用 GQA/MQA 的重要原因之一。

### 2.5 显存计算示例

假设一个 LLaMA 风格模型配置如下：

- \(B = 1\)
- \(T = 4096\)
- \(L = 32\)
- \(H_{kv} = 8\)
- \(D = 128\)
- \(s = 2\)（FP16）

则 KV Cache 大小为：

\[
1 \times 4096 \times 32 \times 8 \times 128 \times 2 \times 2
= 536{,}870{,}912 \text{ bytes}
\]

约等于：

\[
512 \text{ MB}
\]

这意味着在这个配置下，单 batch、4096 上下文长度时，光 KV Cache 就大约需要 512 MB 显存。

如果其他条件不变，但改成普通 MHA，令：

\[
H_{kv} = 32
\]

那么显存会扩大为原来的 4 倍，接近：

\[
2 \text{ GB}
\]

这说明长上下文场景下，KV Cache 很容易成为主要显存瓶颈，而不只是“参数之外的一点小开销”。

## 3. 一条完整的 KV Cache 执行链

把 KV Cache 放回完整推理流程，可以得到这样一条执行链：

1. 输入 prompt：\(x_1,\dots,x_T\)。
2. 进入 `prefill`，一次性处理整段 prompt。
3. 在每一层计算整段历史 token 的 \(K/V\)。
4. 将各层历史 \(K/V\) 写入 cache。
5. 取最后一个位置的 logits：

\[
z_T = W_{\text{vocab}} h_T^{(L)},
\qquad
z_T \in \mathbb{R}^{|\mathcal V|}
\]

6. 根据 \(z_T\) 采样或贪心选择下一个 token \(x_{T+1}\)。
7. 进入 `decode`。
8. 每步只为新 token 计算新的 \(q/k/v\)。
9. 用新 token 的 \(q\) 读取历史 cache 中的 \(K/V\)。
10. 把新 token 的 \(K/V\) 追加到 cache。
11. 重复上述过程，直到生成到 `eos` 或达到长度上限。

这条链里最关键的点只有一个：`prefill` 负责建立缓存，`decode` 负责复用并扩展缓存。

## 4. Prefill 和 Decode 的区别

### 4.1 Prefill 在做什么

设 prompt 长度为 \(T\)，在 `prefill` 中，每层输入是：

\[
H^{(l-1)} \in \mathbb{R}^{T \times d_{\text{model}}}
\]

如果带 batch 和多头，一个常见实现可以把投影后的张量理解为：

\[
Q^{(l)} \in \mathbb{R}^{B \times H_q \times T \times d_h}
\]

\[
K^{(l)}, V^{(l)} \in \mathbb{R}^{B \times H_{kv} \times T \times d_h}
\]

这一层会一次性计算整段输入的 \(Q^{(l)}, K^{(l)}, V^{(l)}\)，并做整段 causal attention。此时模型不只是为了最后一个位置的 logits 在前向，它还顺手建立了每一层、每一个历史 token 的 KV cache。

因此 `prefill` 的本质是：

\[
\boxed{
\text{一次性把整段 prompt 编码进各层 KV cache}
}
\]

### 4.2 Decode 在做什么

当 cache 中已经有长度为 \(T\) 的历史后，下一步生成只需要处理一个新 token。设新 token 为 \(x_{T+1}\)，则第 \(l\) 层只需要新算：

\[
q_{T+1}^{(l)} \in \mathbb{R}^{B \times H_q \times 1 \times d_h}
\]

\[
k_{T+1}^{(l)}, v_{T+1}^{(l)} \in \mathbb{R}^{B \times H_{kv} \times 1 \times d_h}
\]

历史 cache 的形状则是：

\[
K_{\le T}^{(l)}, V_{\le T}^{(l)} \in \mathbb{R}^{B \times H_{kv} \times T \times d_h}
\]

然后把新的 \(k/v\) 追加到历史 cache 后面，再用当前 token 的 \(q\) 去读整段历史 \(K/V\)。

因此 `decode` 的本质是：

\[
\boxed{
\text{每步只计算一个新 token，并复用已有 KV cache}
}
\]

### 4.3 两者在计算复杂度上的区别

`prefill` 处理的是整段长度为 \(T\) 的序列，attention 更接近在计算一个 \(T \times T\) 的关系，因此复杂度更接近：

\[
O(T^2)
\]

`decode` 每一步只处理一个新 token，只需要这个 token 对长度为 \(T\) 的历史做 attention，因此单步复杂度更接近：

\[
O(T)
\]

这也是为什么长 prompt 输入时，`prefill` 会比较贵；而长生成过程中，`decode` 会随着历史越来越长而逐步变慢。

### 4.4 两者在显存行为上的区别

`prefill` 更像是在“写 cache”：

- 一次性生成所有历史 token 的 \(K/V\)
- 把这些 \(K/V\) 放入 cache

`decode` 更像是在“读 cache + 追加 cache”：

- 每步读取已有历史 KV
- 每步只新增一个 token 的 \(K/V\)

因此 KV cache 的显存不会在 `prefill` 后停止增长，而是会随着生成长度继续线性增长：

\[
T \uparrow \quad \Rightarrow \quad \text{KV cache memory} \uparrow
\]

## 5. Prefill 能不能只计算最后一个 Q

直觉上，好像我们只需要预测：

\[
P(x_{T+1} \mid x_1,\dots,x_T)
\]

又因为最终只会取最后一个位置的 logits，所以似乎只需要最后一个位置的 \(q_T\)。这个说法只对了一半。

### 5.1 为什么会觉得“只要最后一个 Q 就够了”

如果只看单层 attention 的最后一个位置，那么它的计算确实是：

\[
o_T^{(l)} =
\sum_{j \le T}\alpha_{Tj}^{(l)} v_j^{(l)}
\]

\[
\alpha_{Tj}^{(l)} =
\operatorname{softmax}_j
\left(
\frac{q_T^{(l)} {k_j^{(l)}}^\top}{\sqrt d_h}
\right)
\]

这里直接出现的只有：

- 最后一个位置自己的 \(q_T^{(l)}\)
- 全部历史位置的 \(K/V\)

所以在“单层、单个位置”的局部公式里，前面位置的 \(q_j^{(l)}\) 的确不直接出现。

### 5.2 真正的问题在于上层的 K/V 从哪里来

要算第 \(l\) 层最后一个位置的输出，你不仅需要：

\[
q_T^{(l)}
\]

还需要：

\[
K_{\le T}^{(l)}, \quad V_{\le T}^{(l)}
\]

但这些 \(K/V\) 不是白来的。它们来自第 \(l-1\) 层所有位置的 hidden states：

\[
k_j^{(l)} = W_K^{(l)} h_j^{(l-1)}, \qquad
v_j^{(l)} = W_V^{(l)} h_j^{(l-1)}
\]

也就是说，为了得到第 \(l\) 层全部历史位置的 \(K/V\)，你必须先得到第 \(l-1\) 层全部历史位置的 hidden states：

\[
H^{(l-1)} \in \mathbb{R}^{T \times d_{\text{model}}}
\]

而这些 hidden states 又来自更低一层的 attention 和 MLP。于是会形成一条完整递推链：

\[
H^{(0)}
\rightarrow H^{(1)}
\rightarrow H^{(2)}
\rightarrow \cdots
\rightarrow H^{(L)}
\]

所以问题不在于“最后一个位置本层是不是只直接用自己的 \(q\)”，而在于：

\[
\boxed{
\text{为了让最后一个位置在高层能读到全部历史 } K/V,\text{ 你必须先把历史位置逐层算出来}
}
\]

### 5.3 更准确的结论

更准确地说：

1. 在某一层内部，算最后一个位置时，前面位置的 \(q\) 不直接进入这一层最后一个位置的 attention 公式。
2. 但在整个多层 Transformer 里，前面位置的 hidden state 必须先被算出来，才能生成上层需要的历史 \(K/V\)。
3. 因此标准的 `prefill` 不能只靠“最后一个位置的 \(q\)”完成，它本质上仍然需要整段 prompt 的完整前向。

也就是说，下面这个说法不成立：

\[
\text{prefill} \equiv \text{只算最后一个 } q
\]

真正成立的是：

\[
\text{prefill} \equiv \text{完整处理整段 prompt，并顺手建立各层历史 } K/V
\]

## 6. 最小实现

```python
import math
import torch
import torch.nn as nn


class TinyAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, past_k=None, past_v=None):
        # x: [B, T_new, D]
        q = self.q_proj(x)  # [B, T_new, D]
        k = self.k_proj(x)  # [B, T_new, D]
        v = self.v_proj(x)  # [B, T_new, D]

        # decode
        if past_k is not None and past_v is not None:
            # past_k/past_v: [B, T_hist, D]
            k_all = torch.cat([past_k, k], dim=1)  # [B, T_hist + T_new, D]
            v_all = torch.cat([past_v, v], dim=1)  # [B, T_hist + T_new, D]
        # prefill
        else:
            k_all = k  # [B, T_new, D]
            v_all = v  # [B, T_new, D]

        scores = torch.matmul(q, k_all.transpose(-1, -2)) / math.sqrt(q.size(-1))
        # scores: [B, T_new, T_hist + T_new]
        attn = torch.softmax(scores, dim=-1)  # [B, T_new, T_hist + T_new]
        out = torch.matmul(attn, v_all)  # [B, T_new, D]

        return self.o_proj(out), k_all, v_all  # out: [B, T_new, D]
```

第一，`prefill` 时，`x` 是整段 prompt，`past_k/past_v` 为空，返回的 `k_all/v_all` 就是初始化后的 cache。

第二，`decode` 时，`x` 通常只包含一个新 token，而 `past_k/past_v` 是历史 cache，新的 `k/v` 会被拼接到末尾。

## 7. 从实现看 KV Cache 的整体流程

### 7.1 cache tensor 在保存什么

在工程实现里，每层 cache 常见的形状是：

\[
[B, H_{kv}, T, D]
\]

其中：

- \(B\)：batch size
- \(H_{kv}\)：KV 头数
- \(T\)：当前缓存长度
- \(D\)：单个 KV head 的维度

也有实现会把维度组织成其他顺序，但本质都一样：每层都在保存 batch 里每个样本、每个 KV head、每个历史位置的 \(K/V\) 向量。

### 7.2 prefill 时 cache 怎么写入

`prefill` 会把整段 prompt 一次性过模型。在每一层中：

1. 计算整段输入的 \(K/V\)
2. 将对应层的 \(K/V\) 作为初始 cache 返回

所以 `prefill` 结束后，cache 里已经有了 prompt 的完整历史。

### 7.3 decode 时 cache 怎么追加

`decode` 每次只输入一个新 token。在每一层中：

1. 计算当前 token 的 \(k/v\)
2. 和历史 cache 进行拼接
3. 用当前 token 的 \(q\) 对整段历史 \(K/V\) 做 attention

因此 decode 不是“重算整段历史”，而是“读取历史 cache，并在末尾追加一个新位置”。

### 7.4 为什么最后只取 `logits[:, -1, :]`

给定输入：

\[
x_1,\dots,x_T
\]

模型会输出每个位置的 logits：

\[
Z \in \mathbb{R}^{T \times |\mathcal V|}
\]

如果带 batch，则常见形状为：

\[
Z \in \mathbb{R}^{B \times T \times |\mathcal V|}
\]

其中最后一个位置的 logits：

\[
z_T \in \mathbb{R}^{|\mathcal V|}
\]

表示的正是：

\[
P(x_{T+1} \mid x_1,\dots,x_T)
\]

因此生成时只需要取最后一个位置，也就是代码里常见的：

```python
next_token_logits = logits[:, -1, :]  # [B, |V|]
```

### 7.5 为什么单个 token 的 logits 仍然需要历史 KV

如果已经拿到了最后一层最后一个位置的 hidden state：

\[
h_T^{(L)} \in \mathbb{R}^{d_{\text{model}}}
\]

那么 logits 只是一个线性投影：

\[
z_T = W_{\text{vocab}} h_T^{(L)},
\qquad
W_{\text{vocab}} \in \mathbb{R}^{|\mathcal V| \times d_{\text{model}}}
\]

这一步本身不需要历史 \(K/V\)。但问题在于，得到 \(h_T^{(L)}\) 之前，每一层 attention 都需要读取历史 \(K/V\)。所以更准确的说法是：

\[
\boxed{
\text{logits 投影本身不需要历史 KV，但 logits 之前的 hidden state 计算需要历史 KV}
}
\]
