# Infra Note Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重写一篇 nano-vLLM KV Cache 笔记，并配套一张可编辑的 draw.io 架构图和网页 SVG 样例。

**Architecture:** 继续使用现有 `note.html?path=infra/nano-vllm/kvcache_and_paged_attention.md` 动态阅读器，不复制 HTML 页面。图形资产以 `.drawio` 为编辑源、`.svg` 为发布文件，Markdown 用相对路径嵌入 SVG；结构测试直接检查 XML、标题层级和内容契约。

**Tech Stack:** Markdown、SVG 1.1、draw.io `mxGraphModel` XML、Python `unittest`、现有 Marked/Highlight.js/KaTeX 阅读器。

## Global Constraints

- 仅修改 `kvcache_and_paged_attention.md`，不改写其他 Infra 笔记。
- 不补写空白的 `点乘_softmax_norm.md`。
- 不在本试点中实现目录自动扫描。
- 桌面端继续同时显示左右导航；窄屏继续沿用现有按需展开行为。
- 不引入新的前端运行时依赖或绘图库。
- 图形使用纯白背景、莫兰迪粉色描边与低饱和辅助色。
- 所有代码围栏使用完整语言名 `python` 或 `text`。
- 技术事实以 nano-vLLM 官方 `block_manager.py`、`model_runner.py` 和 PagedAttention 原始论文为依据。

---

### Task 1: 锁定试点内容与图形契约

**Files:**
- Modify: `tests/test_site_structure.py`
- Create: `docs/assets/infra/nano-vllm/kv-cache-dataflow.drawio`
- Create: `docs/assets/infra/nano-vllm/kv-cache-dataflow.svg`

**Interfaces:**
- Consumes: Python 标准库 `xml.etree.ElementTree`；已确认的资源路径。
- Produces: 可由 draw.io 打开的 `mxfile` 文档，以及 Markdown 可直接加载的 SVG 文件。

- [ ] **Step 1: 写资源契约失败测试**

在 `tests/test_site_structure.py` 顶部加入 `import xml.etree.ElementTree as ET`，并新增：

```python
class InfraNotePilotTests(unittest.TestCase):
    NOTE_PATH = DOCS_DIR / "infra" / "nano-vllm" / "kvcache_and_paged_attention.md"
    DRAWIO_PATH = DOCS_DIR / "assets" / "infra" / "nano-vllm" / "kv-cache-dataflow.drawio"
    SVG_PATH = DOCS_DIR / "assets" / "infra" / "nano-vllm" / "kv-cache-dataflow.svg"

    def test_kv_cache_diagram_has_editable_source_and_accessible_svg(self):
        self.assertTrue(self.DRAWIO_PATH.is_file())
        self.assertTrue(self.SVG_PATH.is_file())

        drawio_root = ET.parse(self.DRAWIO_PATH).getroot()
        self.assertEqual(drawio_root.tag, "mxfile")
        self.assertIsNotNone(drawio_root.find("./diagram/mxGraphModel/root"))

        svg_root = ET.parse(self.SVG_PATH).getroot()
        self.assertEqual(svg_root.attrib.get("viewBox"), "0 0 1200 560")
        svg_text = " ".join(svg_root.itertext())
        for label in (
            "逻辑层",
            "映射层",
            "物理层",
            "BlockManager",
            "block_table",
            "slot_mapping",
            "KV Cache Tensor",
        ):
            self.assertIn(label, svg_text)
```

- [ ] **Step 2: 运行测试并确认红灯**

Run: `.venv\Scripts\python.exe -m unittest tests.test_site_structure.InfraNotePilotTests.test_kv_cache_diagram_has_editable_source_and_accessible_svg -v`

Expected: FAIL，提示 `.drawio` 或 `.svg` 文件不存在。

- [ ] **Step 3: 创建 draw.io 编辑源**

创建 `mxfile > diagram > mxGraphModel > root` XML。根节点固定为 `id="0"` 与 `id="1" parent="0"`，画布 `pageWidth="1200" pageHeight="560"`。创建以下可编辑节点：

```text
logic-layer       逻辑层                     x=40   y=70  w=300 h=390
sequence          Sequence                   x=75   y=145 w=230 h=70
block-manager     BlockManager               x=75   y=290 w=230 h=90
block-table       block_table                x=390  y=290 w=190 h=90
mapping-layer     映射层                     x=620  y=70  w=240 h=390
prepare           prepare_prefill / decode  x=655  y=160 w=170 h=90
slot-mapping      slot_mapping               x=655  y=315 w=170 h=70
physical-layer    物理层                     x=900  y=70  w=260 h=390
store-kvcache     store_kvcache              x=945  y=160 w=170 h=90
kv-cache-tensor   KV Cache Tensor            x=945  y=315 w=170 h=90
```

节点使用 `rounded=1;whiteSpace=wrap;html=1;strokeColor=#b98991;fillColor=#faf5f6;fontColor=#3f3638`。主数据流边使用 `endArrow=block;strokeColor=#8e5f68;strokeWidth=2`，连接顺序为 `Sequence → BlockManager → block_table → prepare → slot_mapping → store_kvcache → KV Cache Tensor`。

- [ ] **Step 4: 创建发布 SVG**

SVG 必须具有以下固定外壳，并用 `<g>`、`<rect>`、`<path>`、`<text>` 绘制与 draw.io 相同的节点和连接关系：

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 560" role="img" aria-labelledby="title desc">
  <title id="title">nano-vLLM KV Cache 数据流</title>
  <desc id="desc">从 Sequence 和 BlockManager 生成 block table，经输入准备映射为 slot mapping，最终写入物理 KV Cache Tensor。</desc>
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
      <path d="M0,0 L10,5 L0,10 Z" fill="#8e5f68"/>
    </marker>
  </defs>
</svg>
```

所有文本节点使用系统中文字体回退 `Inter, "Noto Sans SC", "Microsoft YaHei", sans-serif`；背景为 `#ffffff`，主描边为 `#b98991`，主箭头为 `#8e5f68`，正文为 `#3f3638`。

- [ ] **Step 5: 运行资源测试并确认绿灯**

Run: `.venv\Scripts\python.exe -m unittest tests.test_site_structure.InfraNotePilotTests.test_kv_cache_diagram_has_editable_source_and_accessible_svg -v`

Expected: PASS。

- [ ] **Step 6: 提交图形资产**

```powershell
git add tests/test_site_structure.py docs/assets/infra/nano-vllm
git commit -m "feat: add kv cache dataflow diagram"
```

### Task 2: 重写 KV Cache 试点笔记

**Files:**
- Modify: `tests/test_site_structure.py`
- Modify: `docs/infra/nano-vllm/kvcache_and_paged_attention.md`

**Interfaces:**
- Consumes: `../assets/infra/nano-vllm/kv-cache-dataflow.svg` 相对于 Markdown 的正确路径 `../../assets/infra/nano-vllm/kv-cache-dataflow.svg`。
- Produces: 标题层级连续、代码围栏完整、可由现有阅读器直接渲染的 Markdown。

- [ ] **Step 1: 写笔记内容失败测试**

向 `InfraNotePilotTests` 增加：

```python
    def test_kv_cache_note_has_the_pilot_structure(self):
        markdown = self.NOTE_PATH.read_text(encoding="utf-8")
        self.assertTrue(markdown.startswith("# nano-vLLM：KV Cache 与 Paged Attention\n"))
        self.assertNotIn("---\nid:", markdown)
        self.assertNotIn("把这一段格式变成图", markdown)
        self.assertIn(
            "![nano-vLLM KV Cache 数据流](../../assets/infra/nano-vllm/kv-cache-dataflow.svg)",
            markdown,
        )

        expected_sections = (
            "## 这篇笔记解决什么问题",
            "## 三层数据流",
            "## 全局 KV Cache Tensor",
            "## BlockManager 生命周期",
            "## 从 block_table 到 slot_mapping",
            "## 完整映射示例",
            "## 关键结论与常见误区",
        )
        positions = [markdown.index(section) for section in expected_sections]
        self.assertEqual(positions, sorted(positions))

        for required in (
            "slot = block_id * block_size + offset",
            "can_allocate",
            "allocate",
            "may_append",
            "deallocate",
            "ref_count",
            "完整块",
        ):
            self.assertIn(required, markdown)
```

- [ ] **Step 2: 运行测试并确认红灯**

Run: `.venv\Scripts\python.exe -m unittest tests.test_site_structure.InfraNotePilotTests.test_kv_cache_note_has_the_pilot_structure -v`

Expected: FAIL，首个一级标题或章节契约不满足。

- [ ] **Step 3: 重写标题、摘要与三层架构**

删除 YAML 元信息和待办文本。按契约创建 H1/H2 结构，在“三层数据流”中嵌入 SVG，并用表格解释三层职责：

```markdown
| 层级 | 核心对象 | 输入 | 输出 |
| --- | --- | --- | --- |
| 逻辑层 | `Sequence`、`BlockManager` | token 序列与缓存状态 | `block_table` |
| 映射层 | `prepare_prefill`、`prepare_decode` | `block_table` 与 token 位置 | `slot_mapping` |
| 物理层 | `store_kvcache` | K/V 向量与 `slot_mapping` | KV Cache Tensor 中的物理写入 |
```

- [ ] **Step 4: 重写张量与生命周期章节**

用表格解释 `[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]`。根据官方 `BlockManager` 实现说明：

```python
def can_allocate(seq):
    prefix_hash = -1
    num_cached_blocks = 0
    for logical_block in range(seq.num_blocks - 1):
        token_ids = seq.block(logical_block)
        prefix_hash = compute_hash(token_ids, prefix_hash)
        block_id = hash_to_block_id.get(prefix_hash)
        if block_id is None or blocks[block_id].token_ids != token_ids:
            break
        num_cached_blocks += 1
    return num_cached_blocks

def allocate(seq, num_cached_blocks):
    for logical_block in range(num_cached_blocks):
        block_id = lookup_cached_block(seq.block(logical_block))
        blocks[block_id].ref_count += 1
        seq.block_table.append(block_id)
    for logical_block in range(num_cached_blocks, seq.num_blocks):
        seq.block_table.append(allocate_free_block())
    seq.num_cached_tokens = num_cached_blocks * block_size
```

代码块必须明确写“结构化伪代码”，只用于表达控制流，不声称可直接运行。正文分别解释 `can_allocate`、`allocate`、`may_append`、`hash_blocks` 和 `deallocate`，并纠正“释放后一定仍保留哈希映射”的过度表述：空闲块可以保留内容，但再次分配时若仍是哈希表当前映射，旧映射会被删除。

- [ ] **Step 5: 加入可复算的 slot 映射示例**

固定示例参数：

```text
block_size = 4
token positions = [0, 1, 2, 3, 4, 5]
block_table = [7, 2]
slots = [28, 29, 30, 31, 8, 9]
```

逐步展示 `logical_block = position // block_size`、`offset = position % block_size`、`block_id = block_table[logical_block]` 与 `slot = block_id * block_size + offset`。明确 `slot_mapping` 描述本轮 token 的写入地址，PagedAttention 读取历史 K/V 时仍依赖块表与上下文元数据。

- [ ] **Step 6: 加入结论、误区与来源**

结论覆盖逻辑连续、物理离散、完整块前缀缓存、引用计数和 decode 跨块分配。来源只列官方仓库与原始论文：

```markdown
## 参考

- nano-vLLM：`nanovllm/engine/block_manager.py`
- nano-vLLM：`nanovllm/engine/model_runner.py`
- Kwon 等，*Efficient Memory Management for Large Language Model Serving with PagedAttention*
```

- [ ] **Step 7: 运行内容与围栏测试**

Run: `.venv\Scripts\python.exe -m unittest tests.test_site_structure.InfraNotePilotTests tests.test_site_structure.MarkdownCodeFenceTests -v`

Expected: PASS，且没有无语言、缩写或未闭合围栏。

- [ ] **Step 8: 提交试点笔记**

```powershell
git add tests/test_site_structure.py docs/infra/nano-vllm/kvcache_and_paged_attention.md
git commit -m "docs: rewrite kv cache note"
```

### Task 3: 完整验收与浏览器交付

**Files:**
- Verify: `docs/note.html`
- Verify: `docs/note.css`
- Verify: `docs/infra/nano-vllm/kvcache_and_paged_attention.md`
- Verify: `docs/assets/infra/nano-vllm/kv-cache-dataflow.svg`

**Interfaces:**
- Consumes: 现有本地服务器 `http://127.0.0.1:8000/` 与统一阅读器。
- Produces: 用户可直接查看的 KV Cache 笔记标签页。

- [ ] **Step 1: 运行完整自动化测试**

```powershell
.venv\Scripts\python.exe -m unittest tests.test_site_structure -v
node tests\code-rendering.test.js
node tests\note-paths.test.js
node --check docs\script.js
node --check docs\note-paths.js
node --check docs\code-rendering.js
node --check docs\note.js
git diff --check
```

Expected: Python 测试、Node 测试和语法检查全部退出码为 0，`git diff --check` 无输出。

- [ ] **Step 2: 验证 HTTP 资源**

请求以下 URL，全部必须返回 200：

```text
http://127.0.0.1:8000/note.html?path=infra/nano-vllm/kvcache_and_paged_attention.md
http://127.0.0.1:8000/infra/nano-vllm/kvcache_and_paged_attention.md
http://127.0.0.1:8000/assets/infra/nano-vllm/kv-cache-dataflow.svg
```

- [ ] **Step 3: 浏览器检查桌面布局**

打开试点 URL，验证左侧 nano-vLLM 导航与右侧目录同时可见；正文图形完整显示；代码块显示 Python 标签和 Catppuccin Mocha；页面没有横向滚动。

- [ ] **Step 4: 浏览器检查移动布局**

临时使用 `390 × 844` 视口，验证正文和 SVG 不产生页面级横向滚动，左右导航通过 Notes/Contents 控件按需展开；完成后清除临时视口设置。

- [ ] **Step 5: 将试点页面作为交付页保留**

浏览器最终停留在：

```text
http://127.0.0.1:8000/note.html?path=infra/nano-vllm/kvcache_and_paged_attention.md
```

Git 工作区必须保持干净；若浏览器验收发现问题，先补失败测试，再修复并单独提交。
