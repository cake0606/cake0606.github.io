# Greedy Sampling Connector Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重新布置 Greedy Sampling 图的节点与正交连线，消除连线重叠、穿过节点和箭头吞没短线的问题。

**Architecture:** SVG 继续作为网页发布文件，draw.io 继续作为可编辑源；两者共享相同的节点 ID 和源—目标连接集合。SVG 额外提供非视觉 `data-*` 元数据，Python 结构测试直接解析真实节点边界与连线路径，验证拓扑、避障、非重叠和箭头末段长度。

**Tech Stack:** SVG、draw.io `mxGraphModel` XML、Python 标准库 `unittest` / `xml.etree.ElementTree`、Microsoft Edge 无头渲染。

## Global Constraints

- 只修改 Greedy Sampling 图、对应笔记资源版本和直接回归测试。
- 保留 `viewBox="0 0 1200 700"`、现有文字、配色和无障碍属性。
- SVG 与 draw.io 的节点和源—目标连接集合必须一致。
- 连线不得进入非起止节点内部，不得有正长度的共线重叠。
- 每条箭头路径的最后一段长度必须不少于 36 个 SVG 单位。
- 不引入第三方库、绘图库或自动导出工具。

---

### Task 1: 建立连线几何与拓扑回归测试

**Files:**
- Modify: `tests/test_site_structure.py:55`
- Modify: `tests/test_site_structure.py:544`

**Interfaces:**
- Consumes: SVG 的 `data-node`、`data-edge`、`data-source`、`data-target` 属性，以及 draw.io `mxCell` 的 `vertex`、`edge`、`source`、`target` 属性。
- Produces: `parse_orthogonal_path(path_data) -> list[tuple[float, float]]`、`svg_node_bounds(node) -> tuple[float, float, float, float]`、`segment_enters_bounds(segment, bounds) -> bool`、`collinear_overlap_length(first, second) -> float`，以及三项用户可见行为测试。

- [ ] **Step 1: 写路径与几何测试工具**

在常量区加入：

```python
SVG_NAMESPACE = "{http://www.w3.org/2000/svg}"
SVG_NUMBER_PATTERN = re.compile(r"[MHVL]|-?(?:\d+(?:\.\d*)?|\.\d+)")


def parse_orthogonal_path(path_data):
    tokens = SVG_NUMBER_PATTERN.findall(path_data)
    points = []
    current = None
    index = 0

    while index < len(tokens):
        command = tokens[index]
        index += 1
        if command in ("M", "L"):
            current = (float(tokens[index]), float(tokens[index + 1]))
            index += 2
        elif command == "H":
            current = (float(tokens[index]), current[1])
            index += 1
        elif command == "V":
            current = (current[0], float(tokens[index]))
            index += 1
        else:
            raise ValueError(f"Unsupported SVG path command: {command}")
        points.append(current)

    return points


def svg_node_bounds(node):
    if node.tag == f"{SVG_NAMESPACE}rect":
        x = float(node.attrib["x"])
        y = float(node.attrib["y"])
        return x, y, x + float(node.attrib["width"]), y + float(node.attrib["height"])

    if node.tag == f"{SVG_NAMESPACE}polygon":
        points = [
            tuple(map(float, pair.split(",")))
            for pair in node.attrib["points"].split()
        ]
        xs, ys = zip(*points)
        return min(xs), min(ys), max(xs), max(ys)

    raise AssertionError(f"Unsupported node element: {node.tag}")


def segment_enters_bounds(segment, bounds):
    (x1, y1), (x2, y2) = segment
    left, top, right, bottom = bounds
    if y1 == y2:
        return top < y1 < bottom and max(min(x1, x2), left) < min(max(x1, x2), right)
    if x1 == x2:
        return left < x1 < right and max(min(y1, y2), top) < min(max(y1, y2), bottom)
    raise AssertionError(f"Non-orthogonal segment: {segment}")


def collinear_overlap_length(first, second):
    (x1, y1), (x2, y2) = first
    (x3, y3), (x4, y4) = second
    if y1 == y2 == y3 == y4:
        return max(0, min(max(x1, x2), max(x3, x4)) - max(min(x1, x2), min(x3, x4)))
    if x1 == x2 == x3 == x4:
        return max(0, min(max(y1, y2), max(y3, y4)) - max(min(y1, y2), min(y3, y4)))
    return 0
```

- [ ] **Step 2: 写三个失败测试**

向 `InfraNotePilotTests` 加入 `_greedy_svg_geometry` 测试工具和以下测试：

```python
    def _greedy_svg_geometry(self):
        svg_root = ET.parse(self.VLLM_SAMPLING_SVG_PATH).getroot()
        node_elements = [
            element for element in svg_root.iter() if "data-node" in element.attrib
        ]
        edge_elements = [
            element
            for element in svg_root.iter(f"{SVG_NAMESPACE}path")
            if "edge" in element.attrib.get("class", "").split()
        ]
        self.assertEqual(
            len(node_elements),
            len({element.attrib["data-node"] for element in node_elements}),
        )
        self.assertEqual(
            len(edge_elements),
            len({element.attrib.get("data-edge") for element in edge_elements}),
        )
        nodes = {
            element.attrib["data-node"]: element
            for element in node_elements
        }
        edges = {
            element.attrib["data-edge"]: element
            for element in edge_elements
        }
        return nodes, edges

    def test_greedy_sampling_svg_and_drawio_keep_the_same_topology(self):
        nodes, edges = self._greedy_svg_geometry()
        self.assertTrue(nodes)
        self.assertTrue(edges)

        drawio_root = ET.parse(self.VLLM_SAMPLING_DRAWIO_PATH).getroot()
        cells = drawio_root.findall("./diagram/mxGraphModel/root/mxCell")
        node_cells = [cell for cell in cells if cell.attrib.get("vertex") == "1"]
        edge_cells = [cell for cell in cells if cell.attrib.get("edge") == "1"]
        drawio_nodes = {cell.attrib["id"] for cell in node_cells}
        drawio_edges = {
            (cell.attrib["source"], cell.attrib["target"])
            for cell in edge_cells
        }
        svg_edges = {
            (edge.attrib["data-source"], edge.attrib["data-target"])
            for edge in edges.values()
        }

        self.assertEqual(len(node_cells), len(drawio_nodes))
        self.assertEqual(len(edge_cells), len(drawio_edges))
        self.assertEqual(set(nodes), drawio_nodes)
        self.assertEqual(len(svg_edges), len(edges))
        self.assertEqual(svg_edges, drawio_edges)

    def test_greedy_sampling_connectors_clear_nodes_and_each_other(self):
        nodes, edges = self._greedy_svg_geometry()
        bounds = {node_id: svg_node_bounds(node) for node_id, node in nodes.items()}
        segments = {
            edge_id: list(zip(points, points[1:]))
            for edge_id, edge in edges.items()
            for points in (parse_orthogonal_path(edge.attrib["d"]),)
        }

        for edge_id, edge_segments in segments.items():
            edge = edges[edge_id]
            endpoints = {edge.attrib["data-source"], edge.attrib["data-target"]}
            for node_id, node_bounds in bounds.items():
                if node_id not in endpoints:
                    self.assertFalse(
                        any(segment_enters_bounds(segment, node_bounds) for segment in edge_segments),
                        f"{edge_id} enters {node_id}",
                    )

        edge_ids = list(edges)
        for index, first_id in enumerate(edge_ids):
            for second_id in edge_ids[index + 1:]:
                overlaps = [
                    collinear_overlap_length(first, second)
                    for first in segments[first_id]
                    for second in segments[second_id]
                ]
                self.assertEqual(max(overlaps, default=0), 0, f"{first_id} overlaps {second_id}")

    def test_greedy_sampling_arrow_approaches_are_visible(self):
        _, edges = self._greedy_svg_geometry()
        for edge_id, edge in edges.items():
            points = parse_orthogonal_path(edge.attrib["d"])
            (x1, y1), (x2, y2) = points[-2:]
            self.assertGreaterEqual(
                abs(x2 - x1) + abs(y2 - y1),
                36,
                f"{edge_id} has no visible shaft before its arrow",
            )
```

- [ ] **Step 3: 运行测试并确认红灯**

Run:

```powershell
.venv\Scripts\python.exe -m unittest `
  tests.test_site_structure.InfraNotePilotTests.test_greedy_sampling_svg_and_drawio_keep_the_same_topology `
  tests.test_site_structure.InfraNotePilotTests.test_greedy_sampling_connectors_clear_nodes_and_each_other `
  tests.test_site_structure.InfraNotePilotTests.test_greedy_sampling_arrow_approaches_are_visible -v
```

Expected: 三项测试均 FAIL；首个失败说明 SVG 尚无 `data-node` / `data-edge` 几何契约。

- [ ] **Step 4: 提交失败测试**

```powershell
git add tests/test_site_structure.py
git commit -m "test: cover greedy sampling connector geometry"
```

### Task 2: 同步重排 SVG 与 draw.io

**Files:**
- Modify: `docs/assets/infra/vllm/greedy-sampling-flow.svg`
- Modify: `docs/assets/infra/vllm/greedy-sampling-flow.drawio`

**Interfaces:**
- Consumes: Task 1 的 `data-*` 几何契约和固定 1200 × 700 画布。
- Produces: 19 个同名节点和 24 条同拓扑连线；SVG 边 ID 与 draw.io 边 ID 均为 `e1`–`e24`。

- [ ] **Step 1: 按固定坐标重排节点**

SVG 使用 `rect` 或 `polygon` 绘制并附加 `data-node`；draw.io 使用相同 ID 与包围框：

```text
params            rect     x=35  y=55  w=160 h=56
clamp-check       diamond  x=235 y=38  w=190 h=90
clamp             rect     x=495 y=16  w=175 h=56
verify            rect     x=495 y=122 w=175 h=56
eps               diamond  x=710 y=108 w=190 h=84
greedy            rect     x=950 y=76  w=200 h=56
seed              diamond  x=715 y=228 w=180 h=68
random-seed       rect     x=965 y=202 w=190 h=44
random            rect     x=965 y=266 w=190 h=44
metadata          rect     x=475 y=315 w=250 h=58
all-greedy        rect     x=60  y=430 w=210 h=50
all-random        rect     x=490 y=430 w=220 h=50
mixed             rect     x=920 y=430 w=220 h=50
argmax            rect     x=60  y=520 w=210 h=52
random-pipeline   rect     x=480 y=520 w=240 h=82
mixed-greedy      rect     x=760 y=525 w=180 h=66
mixed-random      rect     x=1000 y=520 w=175 h=86
where             rect     x=900 y=640 w=210 h=50
output            rect     x=490 y=645 w=220 h=45
```

菱形点固定为：

```xml
<polygon data-node="clamp-check" points="330,38 425,83 330,128 235,83" class="accent" />
<polygon data-node="eps" points="805,108 900,150 805,192 710,150" class="accent" />
<polygon data-node="seed" points="805,228 895,262 805,296 715,262" class="node" />
```

- [ ] **Step 2: 用独立轨道重写 24 条 SVG 路径**

每条路径使用 `class="edge"` 以及对应的 `data-edge`、`data-source`、`data-target`；例如第一条为 `<path d="M195 83 H235" class="edge" data-edge="e1" data-source="params" data-target="clamp-check" />`。全部路径固定为：

```text
e1  params→clamp-check          M195 83 H235
e2  clamp-check→clamp          M425 83 H445 V44 H495
e3  clamp-check→verify         M330 128 V150 H495
e4  clamp→verify               M582.5 72 V122
e5  verify→eps                 M670 150 H710
e6  eps→greedy                 M900 150 H910 V104 H950
e7  eps→seed                   M805 192 V228
e8  seed→random-seed           M850 245 H925 V224 H965
e9  seed→random                M850 279 H925 V288 H965
e10 greedy→metadata            M1150 104 H1195 V362 H725
e11 random-seed→metadata       M1155 224 H1185 V344 H725
e12 random→metadata            M1155 288 H1175 V326 H725
e13 metadata→all-greedy        M520 373 V394 H165 V430
e14 metadata→all-random        M600 373 V430
e15 metadata→mixed             M680 373 V394 H1030 V430
e16 all-greedy→argmax          M165 480 V520
e17 all-random→random-pipeline M600 480 V520
e18 mixed→mixed-greedy         M970 480 V485 H850 V525
e19 mixed→mixed-random         M1090 480 V520
e20 mixed-greedy→where         M850 591 V660 H900
e21 mixed-random→where         M1088 606 H1165 V670 H1110
e22 argmax→output              M165 572 V667 H490
e23 random-pipeline→output     M600 602 V645
e24 where→output               M900 675 H760 V667 H710
```

将分隔线移动到 `y=395`、执行区标题移动到 `y=418`。保持所有现有文字，仅把 draw.io 原来的 `mixed-pipeline` 拆成 `mixed-greedy` 与 `mixed-random`，与 SVG 当前已经表达的两条计算分支一致。

- [ ] **Step 3: 同步 draw.io 节点、边和显式折点**

将 `modified` 更新为 `2026-08-11T00:00:00.000Z`。边连接集合严格使用 Step 2 的 24 对源—目标 ID。直线边设置对应 `exitX` / `exitY` / `entryX` / `entryY`；折线边在 `mxGeometry relative="1"` 中加入精确的 `mxPoint` 数组。例如 `e10` 使用：

```xml
<Array as="points">
  <mxPoint x="1195" y="104" />
  <mxPoint x="1195" y="362" />
</Array>
```

其余折线路径使用以下折点；未列出的路径保持直线：

```text
e2  (445,83), (445,44)
e3  (330,150)
e6  (910,150), (910,104)
e8  (925,245), (925,224)
e9  (925,279), (925,288)
e11 (1185,224), (1185,344)
e12 (1175,288), (1175,326)
e13 (520,394), (165,394)
e15 (680,394), (1030,394)
e18 (970,485), (850,485)
e20 (850,660)
e21 (1165,606), (1165,670)
e22 (165,667)
e24 (760,675), (760,667)
```

- [ ] **Step 4: 运行几何测试并确认绿灯**

Run:

```powershell
.venv\Scripts\python.exe -m unittest `
  tests.test_site_structure.InfraNotePilotTests.test_greedy_sampling_svg_and_drawio_keep_the_same_topology `
  tests.test_site_structure.InfraNotePilotTests.test_greedy_sampling_connectors_clear_nodes_and_each_other `
  tests.test_site_structure.InfraNotePilotTests.test_greedy_sampling_arrow_approaches_are_visible `
  tests.test_site_structure.InfraNotePilotTests.test_greedy_sampling_diagram_has_editable_source_and_accessible_svg -v
```

Expected: 4 tests PASS。

- [ ] **Step 5: 提交图形修复**

```powershell
git add docs/assets/infra/vllm/greedy-sampling-flow.svg docs/assets/infra/vllm/greedy-sampling-flow.drawio
git commit -m "fix: reroute greedy sampling connectors"
```

### Task 3: 刷新笔记资源版本

**Files:**
- Modify: `tests/test_site_structure.py:578`
- Modify: `docs/infra/vllm/sampling.md`

**Interfaces:**
- Consumes: 新 SVG 文件。
- Produces: `greedy-sampling-flow.svg?v=20260811-1` 的笔记引用和同步结构契约。

- [ ] **Step 1: 先更新结构测试期望**

把 `test_sampling_note_embeds_the_greedy_flow` 中的期望改为：

```python
"![vLLM Greedy Sampling 流程](../../assets/infra/vllm/greedy-sampling-flow.svg?v=20260811-1)"
```

- [ ] **Step 2: 运行测试并确认红灯**

Run:

```powershell
.venv\Scripts\python.exe -m unittest tests.test_site_structure.InfraNotePilotTests.test_sampling_note_embeds_the_greedy_flow -v
```

Expected: FAIL，显示笔记仍引用 `v=20260810-3`。

- [ ] **Step 3: 更新 Markdown 资源版本**

把 `docs/infra/vllm/sampling.md` 中唯一的图形引用查询参数更新为 `v=20260811-1`，不改正文。

- [ ] **Step 4: 运行测试并确认绿灯**

Run:

```powershell
.venv\Scripts\python.exe -m unittest tests.test_site_structure.InfraNotePilotTests.test_sampling_note_embeds_the_greedy_flow -v
```

Expected: PASS。

- [ ] **Step 5: 提交缓存刷新**

```powershell
git add tests/test_site_structure.py docs/infra/vllm/sampling.md
git commit -m "fix: refresh greedy sampling diagram"
```

### Task 4: 浏览器渲染与完整验证

**Files:**
- Verify: `docs/assets/infra/vllm/greedy-sampling-flow.svg`
- Verify: `docs/assets/infra/vllm/greedy-sampling-flow.drawio`
- Verify: `docs/infra/vllm/sampling.md`
- Verify: `tests/test_site_structure.py`

**Interfaces:**
- Consumes: 完成后的 SVG、draw.io、Markdown 和测试。
- Produces: 原始尺寸与正文缩放尺寸的视觉证据，以及全量测试结果。

- [ ] **Step 1: 用 Edge 渲染两个尺寸**

使用独立临时用户目录渲染：

```powershell
$edge = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
$reviewDir = Join-Path $env:TEMP 'cakeman-greedy-review'
$profileDir = Join-Path $env:TEMP 'cakeman-greedy-edge-profile'
New-Item -ItemType Directory -Force -Path $reviewDir,$profileDir | Out-Null
& $edge --headless --disable-gpu --hide-scrollbars --user-data-dir="$profileDir" --window-size=1200,700 --screenshot="$reviewDir\greedy-1200.png" 'file:///E:/code/cakeman.github.io/docs/assets/infra/vllm/greedy-sampling-flow.svg'
& $edge --headless --disable-gpu --hide-scrollbars --user-data-dir="$profileDir-small" --window-size=820,478 --screenshot="$reviewDir\greedy-820.png" 'file:///E:/code/cakeman.github.io/docs/assets/infra/vllm/greedy-sampling-flow.svg'
```

打开两张 PNG，确认所有箭头有可见线身、无穿框、无重叠，并且标签对应明确。

- [ ] **Step 2: 运行完整自动化验证**

Run:

```powershell
.venv\Scripts\python.exe -m unittest tests.test_site_structure -v
node tests\code-rendering.test.js
node tests\note-paths.test.js
node tests\note-runtime.test.js
node --check docs\script.js
node --check docs\note-paths.js
node --check docs\code-rendering.js
node --check docs\note.js
git diff --check
git status --short
```

Expected: Python 与 Node 测试全部通过；JavaScript 语法检查退出码为 0；`git diff --check` 无输出；工作区无未提交文件。

- [ ] **Step 3: 对照规格逐项验收**

重新读取设计规格，确认四个范围文件之外没有实现改动，19 个节点、24 条边、36 单位末段、资源版本和视觉验收全部满足。
