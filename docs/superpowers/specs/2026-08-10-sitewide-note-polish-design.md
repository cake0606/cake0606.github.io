# 全站笔记与阅读界面统一设计

## 目标

统一个人网站阅读界面和已发布笔记的表达方式：页面使用纯白与中性浅灰作为基础表面，莫兰迪粉仅作为强调色；品牌标识从 `JZ` 简化为 `Z`；代码块自动消除来自原项目的公共前导缩进；所有已发布的非空笔记使用具体的问题标题、行内代码标记和必要的流程图。

## 范围

本次处理首页与四个阅读器入口，以及网站目录中的 18 篇 Markdown 笔记：

- Infra / CUDA：4 篇。
- Infra / nano-vLLM：6 篇。
- Infra / vLLM：2 篇。
- LLM / RL：2 篇。
- LLM / Concepts：4 篇。

空白的 `点乘_softmax_norm.md`、`concepts.md`、`gae.md`、`index.md` 保持空白，不凭空补写技术内容。未发布的 `docs/learn.md`、`docs/elementwise.md` 与 `docs/code-agent/` 不属于本次网站笔记目录，不做内容改写。

## 视觉规则

- 页面与卡片基础表面使用纯白。
- 次级背景、表头、引用块和悬停状态使用中性浅灰，不使用淡粉色铺底。
- 莫兰迪粉保留在边框、焦点、活动状态、行内代码和小范围装饰上。
- 首页、根阅读器和三个兼容阅读器左上角标识统一为 `Z`。
- 不恢复深浅色切换。

## 代码呈现

阅读器在语法高亮前对每个代码块执行公共缩进去除：忽略空行，计算所有非空行共有的最小前导空白并整体删除，同时保留函数体、条件分支和嵌套结构的相对缩进。

该行为属于通用渲染能力，不依赖人工逐篇移动代码。语言标签和 Catppuccin Mocha 主题保持不变。

## 笔记改写规则

- 每篇非空笔记必须有且只有一个明确的 H1。
- 不使用“这篇笔记解决什么问题”一类元叙述；需要问题章节时，使用“KV Cache 与 Paged Attention 解决的问题”“PPO 解决的问题”等具体标题。
- 正文中出现的函数、方法、类、字段、变量、参数、枚举值和短表达式使用行内代码标记。
- 修正明显的大小写、空格、标题层级、围栏和表达顺序问题，但不改变笔记的技术主题。
- 只在执行链、调度链、数据流或分支决策仅靠段落难以理解时增加流程图。
- 新图使用纯白背景、莫兰迪粉描边和中性文字，同时保存 `.drawio` 编辑源与 `.svg` 发布文件。

## vLLM Sampling 流程图

`infra/vllm/sampling.md` 增加 Greedy Sampling 流程图，至少覆盖：

1. `temperature < _SAMPLING_EPS` 的 Greedy 判定。
2. Greedy 参数规整：`top_p = 1.0`、`top_k = 0`、`min_p = 0.0`。
3. Batch 元数据的 `all_greedy`、`all_random` 与混合分支。
4. 混合 Batch 中预先计算 `argmax`，Random 请求继续经过 temperature、processor、top-k/top-p 和采样。
5. 最终通过 `torch.where` 合并 Greedy 与 Random 结果。

文件位置：

- `docs/assets/infra/vllm/greedy-sampling-flow.drawio`
- `docs/assets/infra/vllm/greedy-sampling-flow.svg`

## 并行执行边界

- 子代理 A：仅修改 `docs/infra/nano-vllm/` 及对应新资产目录。
- 子代理 B：仅修改 `docs/infra/cuda/` 及对应新资产目录。
- 子代理 C：仅修改 `docs/llm/rl/`、`docs/llm/concepts/` 及对应新资产目录。
- 主会话：修改 UI、阅读器、测试、`docs/infra/vllm/` 和 vLLM 图形资产。

代理不修改共享测试文件，不创建提交；主会话统一审查、测试和提交。

## 验收标准

- 所有入口的左上角标识均为 `Z`，页面基础背景不再使用淡粉色。
- 代码块公共缩进被删除，内部相对缩进保持正确。
- 14 篇非空已发布笔记均具有 H1、完整围栏和一致标题层级。
- 不再出现“这篇笔记解决什么问题”。
- 关键函数和变量在正文中使用行内代码标记。
- Greedy Sampling 的 draw.io 与 SVG 存在、可编辑、可访问并在笔记中正确显示。
- 空白笔记仍为空白。
- 自动化测试、脚本语法、HTTP 资源和桌面/移动浏览器验收全部通过。
