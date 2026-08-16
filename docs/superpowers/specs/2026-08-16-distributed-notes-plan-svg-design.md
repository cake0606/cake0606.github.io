# Distributed 笔记、Plan 与 SVG 修正设计

## 目标

把新同步的分布式训练笔记完整接入静态网站，在首页 Infra 笔记树中增加 Distributed 目录；更新 Infra Plan，使 DP 与 DDP 显示为已完成、DeepSpeed 显示为未完成；修正 DDP 流程图下半部分的文字、虚线和箭头重叠，并验证两张分布式 SVG 的结构与渲染结果。

## 当前状态

网站直接发布 `docs/`，没有构建期目录扫描。首页笔记树由 `docs/index.html` 手工维护，阅读器侧边栏集合由 `docs/note-paths.js` 手工维护，`tests/test_site_structure.py` 用固定路径清单保证三处一致。

新目录 `docs/infra/distributed/` 已包含以下文件：

- `基础.md`：有完整内容。
- `DP-DDP.md`：有完整内容，并引用两张 SVG。
- `deepspped.md`：空文件，文件名拼写错误。
- `tp-ep-pp.md`：目前只有标题。

两张 SVG 均能解析且具有 `title`、`desc` 和有效 `viewBox`。`ring-all-reduce-bucket.svg` 的当前渲染没有明显问题；`ddp-bucket-overlap.svg` 下半部分的通信重叠说明与 Bucket 1 的 hook、连线互相压叠，需要调整。

## 采用方案

采用完整目录接入方案，继续沿用网站现有的显式静态登记方式。该方案不会引入目录扫描或新的运行时逻辑，能够保持首页、阅读器侧边栏和结构测试对同一笔记集合的明确约束。

不采用只发布 `DP-DDP.md` 的方案，因为它会让磁盘目录与网站导航不一致。不采用自动扫描方案，因为静态 GitHub Pages 页面没有现成的目录清单接口，为本次内容更新增加生成步骤或运行时清单没有必要。

## 文件与导航

将 `docs/infra/distributed/deepspped.md` 重命名为 `docs/infra/distributed/deepspeed.md`，保留空内容，避免网站公开错误拼写。

首页 Infra 的二级目录顺序调整为：

1. CUDA
2. Distributed
3. nano-vLLM
4. vLLM

Distributed 目录按以下顺序显示文件名链接：

1. `基础.md`
2. `DP-DDP.md`
3. `deepspeed.md`
4. `tp-ep-pp.md`

`docs/note-paths.js` 增加 `infra/distributed` 集合，阅读器侧边栏使用以下标题：

| 标题 | 路径 |
| --- | --- |
| Distributed Basics | `infra/distributed/基础.md` |
| DP & DDP | `infra/distributed/DP-DDP.md` |
| DeepSpeed | `infra/distributed/deepspeed.md` |
| TP, EP & PP | `infra/distributed/tp-ep-pp.md` |

## Plan 数据

`docs/plan.json` 仍是 Plan 的唯一数据源。在 Infra 分组的 `triton` 之后、现有 `tp、pp、ep` 之前插入：

```json
{ "text": "DP、DDP", "completed": true },
{ "text": "DeepSpeed", "completed": false }
```

现有 `tp、pp、ep` 保持未完成。其他 Infra 和 LLM 条目的文本、顺序与状态不变。

## DDP SVG 修正

只修改 `docs/assets/infra/distributed/ddp-bucket-overlap.svg` 的下半部分布局，不改变图示含义、配色、画布大小或无障碍文本。

布局调整需要满足：

- `hook：W₁、b₁ ready` 保持与 Backward Layer 1 和 Gradient Bucket 1 的因果连线清晰。
- “通信 Bucket 0 时，Backward Layer 1 仍可继续”作为独立说明，不与 hook 文本、Bucket 1、箭头或虚线相交。
- Bucket 0 与 Bucket 1 的通信顺序仍由现有主流程箭头表达。
- `optimizer.step()` 继续位于全部 bucket 完成之后。
- `ring-all-reduce-bucket.svg` 不做内容修改；它只参与验证。

## 错误处理与兼容性

本次不改变运行时错误处理。笔记仍通过现有 `note.html?path=...` 路由加载，Plan 仍通过现有 JSON 校验和非致命错误状态渲染。所有新增路径使用正斜杠，并继续满足阅读器的路径安全规则。

## 测试与验证

先更新测试期望，再修改内容：

- `tests/test_site_structure.py` 增加四个 Distributed 笔记路径、Distributed 首页目录、Plan 新条目和完成状态断言，并将 `deepspeed.md` 纳入允许为空的笔记清单。
- `tests/note-paths.test.js` 断言 `DP-DDP.md` 能返回完整的 Distributed sibling collection，并验证其 SVG 相对路径解析结果。
- 为两张分布式 SVG 增加 XML 可解析、`title`/`desc`/`viewBox` 存在、内部引用有效的结构检查。
- 为 DDP SVG 的重叠说明区域增加针对性几何断言，防止文字和关键节点再次压叠。
- 运行 Python 与 Node 全部测试。
- 使用本地 SVG 栅格化在 1400×860 与 1400×820 的原始画布尺寸检查两张图，确认无裁切、文字重叠或错误连线。

## 验收标准

1. 首页 Notes/Infra 下出现 Distributed，并可打开四篇笔记。
2. 打开任一 Distributed 笔记时，阅读器侧边栏列出相同的四篇 sibling 笔记。
3. Infra Plan 显示已完成的 `DP、DDP` 和未完成的 `DeepSpeed`，原有 `tp、pp、ep` 仍未完成。
4. DDP SVG 下半部分不再发生文字、虚线和箭头重叠。
5. Ring All-Reduce SVG 保持原有内容且通过结构和渲染检查。
6. 仓库不再包含 `deepspped.md`，所有新增页面链接均指向实际存在的 Markdown 文件。
7. Python 与 Node 测试全部通过。
