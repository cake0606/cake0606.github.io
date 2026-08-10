# Greedy Sampling 连线修复设计

## 目标

修复 Greedy Sampling 矢量图中的三类连线缺陷：连线相互重叠、连线穿过矩形或菱形节点，以及相邻节点过近导致箭头占满整条连线。修复后继续保留现有流程语义、文字内容、配色、1200 × 700 画布与 SVG 无障碍信息。

## 范围

本次只修改 Greedy Sampling 图及其直接引用和回归测试：

- `docs/assets/infra/vllm/greedy-sampling-flow.svg`
- `docs/assets/infra/vllm/greedy-sampling-flow.drawio`
- `docs/infra/vllm/sampling.md`
- `tests/test_site_structure.py`

KV Cache 图、其他笔记、阅读器代码和全站样式均不修改。

## 根因

当前 SVG 使用手写绝对坐标。新增参数规整分支后，部分相邻节点只保留 11–26 个 SVG 单位，而箭头标记在 `markerUnits="strokeWidth"` 和 2 单位描边下约占 18 个单位，箭头因此遮住了大部分线身。

`GREEDY`、`RANDOM_SEED` 与 `RANDOM` 到 `Batch metadata` 的三条路径沿相同的纵向坐标回流，其中 `GREEDY` 路径直接穿过两个 Random 节点，路径之间也共享了长线段。`Batch metadata` 到三个执行分支的路径同样共用主干，无法清楚区分每条连接。

## 布局与路由设计

采用网格化重排，而不是只修改个别折点：

1. 参数规整区仍按从左到右、再向下的阅读顺序排列。
2. 拉开 `_verify_args`、`temperature < _SAMPLING_EPS ?` 与 `seed is set ?`，使每条短连接的箭头前末段至少有 36 个 SVG 单位。
3. `GREEDY`、`RANDOM_SEED` 与 `RANDOM` 分别从独立端口离开节点，沿画布右侧三条互不重合的轨道进入 `Batch metadata` 的不同端口；轨道不得经过任何节点内部。
4. `Batch metadata` 从三个不同底部端口分别连接 `all_greedy`、`all_random` 与 `mixed batch`，不共享线段。
5. 执行区保持三列结构。`mixed batch` 的两条子路径和到 `torch.where` 的两条输入继续使用独立端口与轨道。
6. 不改变任何分支标签、节点含义或最终汇合关系。

SVG 中为节点和连线增加稳定的 `data-node`、`data-edge`、`data-source` 与 `data-target` 元数据。这些属性不改变呈现，用于让测试从实际几何和拓扑验证行为。矩形与菱形的可视几何仍由 SVG 元素本身定义，不另建只供测试使用的隐藏图形。

## 编辑源同步

draw.io 文件继续作为可编辑源，节点位置和连接关系与 SVG 一致。所有边保持正交路由，并设置显式入口、出口或折点，避免 draw.io 再次自动选择会穿过节点的最短路径。

SVG 与 draw.io 的连接集合必须一致；每条连接以源节点 ID 和目标节点 ID 标识。SVG 的精确折点可以针对浏览器渲染微调，但不得改变拓扑。

## 回归测试

扩展现有 `InfraNotePilotTests`，从 XML 解析真实元素并验证：

- 每个 SVG 流程节点和每条连线都有唯一标识。
- SVG 与 draw.io 的源节点、目标节点连接集合相同。
- 任意连线线段不进入非起止节点的包围区域。
- 任意两条连线不存在正长度的共线重叠。
- 每条带箭头路径的最后一段长度不少于 36 个 SVG 单位。
- 现有画布尺寸、无障碍属性和关键文字契约继续成立。

测试只处理本图采用的 `M`、`H`、`V`、`L` 正交路径命令；不引入第三方几何库。

## 发布与视觉验收

笔记中的 SVG 查询参数更新为 `v=20260811-1`，确保浏览器不复用旧缓存，并同步更新结构测试期望。

完成自动化测试后，用真实浏览器分别按 1200 × 700 原始尺寸和笔记正文中的缩放尺寸渲染 SVG。视觉验收确认：

- 每条箭头都有可辨认的线身。
- 连线不穿过任何矩形或菱形。
- 并行路径之间有清晰间隔。
- 分支标签与箭头对应关系明确。
- 文字、配色、画布边界和两段流程的阅读顺序保持不变。

## 非目标

- 不改变 Greedy Sampling 的技术内容。
- 不重新设计全站图形风格。
- 不新增图形运行时、构建依赖或自动 draw.io 导出工具。
- 不修复或调整 KV Cache 图。
