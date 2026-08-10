# vLLM Sampling 笔记整合设计

## 目标

将当前分散在 `sampling.md` 和 `scheduler.md` 中的 vLLM 采样内容整合为一篇层级清晰的 `sampling.md`。文章先统一解释 Greedy、Random 及随机采样过滤参数，再展示 Greedy 与 Random 的完整执行流程，最后介绍 Parallel Sampling 与 Beam Search 这两类多序列生成方式。

## 内容边界

- 单步 token 选择只区分 Greedy 与 Random。
- Temperature、Top-k、Top-p 和 Min-p 是 Random Sampling 的分布调整或候选过滤参数，不作为与 Greedy、Random 并列的策略。
- Parallel Sampling 与 Beam Search 归入“多序列生成”。
- Speculative Decoding 属于解码加速，不纳入本次 Sampling 正文，也不创建空章节。
- 不保留独立的策略对比、关键结论或参考资料章节。

## 文章结构

```text
# vLLM Sampling

## Greedy 与 Random

[解释 Greedy、Random、Temperature、Top-k、Top-p 和 Min-p 的概念与关系]

[Greedy 与 Random Sampling 统一流程图]

### 参数归一化与类型判定

### Batch 聚合与执行路径
#### All Greedy
#### All Random
#### 混合 Batch

### Logits 处理顺序

### Random 候选过滤
#### Top-k
#### Top-p

## 多序列生成

### Parallel Sampling
#### ParentRequest 的职责
#### 执行流程

### Beam Search
#### BeamSearchInstance
#### 每轮扩展流程
```

Greedy 与 Random 不再各自占用三级标题。两者共享同一套参数归一化、batch 分流和 sampler 执行主线，文章按执行阶段展开。

## 现有内容迁移

`sampling.md` 开头的“Sampling 解决的问题”改写为 Greedy 与 Random 的概念说明。原 Greedy 流程图移动到概念说明之后，使读者先理解术语，再阅读完整执行路径。

原“关键结论”中的信息不删除，而是就地归入对应章节：

- `temperature = 0`、`_MAX_TEMP`、`_SAMPLING_EPS` 以及 Greedy 参数归一化进入“参数归一化与类型判定”。
- `all_greedy`、`all_random`、安全温度和 `torch.where` 进入“Batch 聚合与执行路径”。
- processor 是否保持 `argmax` 不变以及相应执行顺序进入“Logits 处理顺序”。
- Top-k 和 Top-p 的实现进入“Random 候选过滤”。

`scheduler.md` 中的 Parallel Sampling 与 Beam Search 正文迁入“多序列生成”。原对比表和关键结论章节删除；其中关于独立 child、累计 beam score、每轮裁剪和输出聚合的有效信息放回对应小节。

## 流程图

现有流程图的节点和连线已经覆盖 Greedy、Random 与混合 batch，无需重新设计几何布局。为匹配扩大的语义范围：

- 将 Draw.io 和 SVG 文件重命名为 `greedy-random-sampling-flow.drawio` 与 `greedy-random-sampling-flow.svg`。
- 将图内可见标题、SVG `<title>`、描述文本和 Markdown 图片替代文本统一改为“vLLM Greedy 与 Random Sampling 流程”。
- 保持 Draw.io 源文件和导出 SVG 的节点、边 ID 同步，并继续使用版本查询参数避免浏览器缓存旧资源。

## 仓库清理

- 删除 `docs/infra/vllm/scheduler.md`。
- 从 `docs/note-paths.js` 和 `docs/index.html` 移除 Scheduler 入口。
- 从站点结构测试的期望路径中移除 `infra/vllm/scheduler.md`。
- 更新 Sampling 笔记和流程图测试，使其断言新标题、资源名、章节层级、多序列内容以及旧文件不存在。

## 验证

- 运行 Python 站点结构测试，验证 Markdown、导航、SVG/Draw.io 同步和旧路径清理。
- 运行 Node 测试，验证笔记路径注册和前端行为未回归。
- 检查 JavaScript 语法。
- 在本地浏览器打开 Sampling 笔记，确认目录只显示新的章节层级、流程图正常加载且 Scheduler 不再出现在导航中。

## 完成标准

- `sampling.md` 只有“Greedy 与 Random”和“多序列生成”两个二级正文主题。
- 概念说明先于统一流程图，Top-k/Top-p 明确归属于 Random Sampling 的候选过滤过程。
- Parallel Sampling 与 Beam Search 内容完整合并。
- 不存在策略对比、统一结论、参考资料或 Speculative Decoding 章节。
- `scheduler.md`、旧导航入口和旧资源引用全部移除。
- 自动化测试和本地页面检查全部通过。
