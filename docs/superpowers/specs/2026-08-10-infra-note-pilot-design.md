# Infra 笔记单篇试点设计

## 目标

先重构 `docs/infra/nano-vllm/kvcache_and_paged_attention.md`，用一篇实际笔记验证后续 Infra 内容整理与图形化表达方式。试点完成后直接打开该笔记供用户查看，再决定是否将方案推广到其余 Infra 笔记。

## 本次范围

- 重写现有 KV Cache 与 Paged Attention 笔记的结构和表达。
- 保留有价值的实现代码，删除残缺代码、待办文本与重复描述。
- 新增一张 draw.io 风格的矢量架构图。
- 同时保存可编辑的 `.drawio` 源文件与网页使用的 `.svg` 文件。
- 继续使用现有统一三栏阅读器；桌面端左右导航始终显示，窄屏沿用按需展开行为。
- 完成后打开该笔记页面供用户验收。

本次不修改空白的 `点乘_softmax_norm.md`，不改写其他 Infra 笔记，也不在试点阶段实现目录自动扫描。

## 笔记内容结构

笔记使用一个明确的一级标题：`nano-vLLM：KV Cache 与 Paged Attention`。

正文按以下顺序组织：

1. 这篇笔记解决什么问题。
2. 从逻辑块到物理 KV Cache 的三层架构。
3. 全局 KV Cache 张量的形状与各维度含义。
4. BlockManager 的 `allocate`、`append`、`deallocate` 生命周期。
5. `block_table` 到 `slot_mapping` 的映射规则。
6. 一组从 token、逻辑块、物理块到最终 slot 的完整示例。
7. 关键结论与常见误区。

代码片段统一使用 `python` 或 `text` 语言标识。示例代码明确标注为伪代码或摘录，避免让不完整片段看起来可以直接运行。

## 架构图

图中使用横向三层数据流：

1. 逻辑层：`Sequence` 与 `BlockManager`，输出 `block_table`。
2. 映射层：`prepare_prefill` / `prepare_decode`，计算 `slot_mapping`。
3. 物理层：`store_kvcache`，把 K/V 写入全局 KV Cache Tensor。

图形使用纯白背景、莫兰迪粉色描边与低饱和辅助色，保持与网站浅色主题一致。节点使用圆角矩形，主数据流使用实线箭头，辅助说明使用虚线连接。SVG 必须包含可缩放的 `viewBox`、可读的文本对比度和简短替代文本。

文件位置：

- `docs/assets/infra/nano-vllm/kv-cache-dataflow.drawio`
- `docs/assets/infra/nano-vllm/kv-cache-dataflow.svg`

Markdown 通过相对路径嵌入 SVG，图下提供一句文字说明，使图片不可用时正文仍然完整。

## 阅读器适配

试点不复制独立 HTML。页面继续通过 `note.html?path=infra/nano-vllm/kvcache_and_paged_attention.md` 加载 Markdown。

现有阅读器保持三栏职责：

- 左栏展示 nano-vLLM 同目录笔记。
- 中栏展示正文、架构图、代码与表格。
- 右栏根据二至四级标题生成目录。

如果 SVG 在正文宽度内显示，CSS 负责限制最大宽度并保持纵横比；不会引入新的绘图库或运行时依赖。

## 验收标准

- 页面不再显示 YAML 元信息或原有待办文字。
- 标题层级从一个 H1 开始，后续层级连续。
- 架构图不是 ASCII 字符画，可在浏览器中清晰缩放。
- `.drawio` 源文件可被 draw.io 打开并继续编辑。
- SVG 与正文在桌面和移动宽度下都不产生页面级横向滚动。
- 代码块继续使用 Catppuccin Mocha，并显示正确的语言标签。
- 左右导航在桌面端始终可见。
- 自动化测试覆盖图片路径、标题结构、残留待办文本和页面资源可访问性。

## 后续推广

用户确认单篇试点后，再单独规划：

- 其余非空 Infra 笔记的分批改写。
- 空白笔记的展示策略。
- 从目录自动生成首页树、左侧笔记导航与清单的构建脚本。
