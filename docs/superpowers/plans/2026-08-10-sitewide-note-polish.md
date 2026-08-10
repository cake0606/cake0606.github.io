# Sitewide Note Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 统一已发布笔记的阅读界面、代码缩进、技术标记、标题表达和必要流程图。

**Architecture:** 通用视觉与代码缩进放在共享 CSS/JavaScript 中一次解决；笔记内容按互不重叠的目录并行整理。流程图使用 `.drawio` 编辑源和 `.svg` 发布文件，Markdown 继续由统一阅读器动态加载。

**Tech Stack:** HTML、CSS、JavaScript、Marked、Highlight.js、Markdown、SVG、draw.io `mxGraphModel`、Python `unittest`、Node `node:test`。

## Global Constraints

- 页面与卡片基础表面使用纯白；次级背景使用中性浅灰，不使用淡粉色铺底。
- 莫兰迪粉只用于边框、焦点、活动状态、行内代码和小范围装饰。
- 首页与四个阅读器入口的品牌标识统一为 `Z`。
- 代码块去除所有非空行共有的最小前导空白，并保留内部相对缩进。
- 14 篇非空已发布笔记必须有且只有一个 H1；4 篇空白笔记保持空白。
- 不出现“这篇笔记解决什么问题”；问题章节使用具体主题名。
- 技术函数、类、变量、参数、枚举和短表达式在正文中使用行内代码。
- 新流程图同时保存 `.drawio` 与 `.svg`，使用白底、莫兰迪粉描边和中性文字。
- 不引入新的前端运行时依赖，不恢复深浅色切换，不改写未发布草稿。

---

### Task 1: 建立全站契约测试

**Files:**
- Modify: `tests/test_site_structure.py`
- Modify: `tests/code-rendering.test.js`

**Interfaces:**
- Consumes: `EXPECTED_NOTE_PATHS` 与现有 HTML 解析器。
- Produces: 全站品牌、背景、H1、空白笔记、具体问题标题和代码缩进契约。

- [ ] **Step 1: 写失败的结构测试**

在 Python 测试中增加：所有入口 `.brand-mark` 文本为 `Z`；样式中的 `--bg-soft` 与 `--accent-mist` 为中性值；14 篇非空笔记各有一个 H1；空白笔记仍为空；所有已发布笔记不包含“这篇笔记解决什么问题”。

- [ ] **Step 2: 写失败的代码缩进测试**

在 Node 测试中要求：

```javascript
assert.equal(
  CodeRendering.dedentCode("    def run():\n        return value\n"),
  "def run():\n    return value\n"
);
```

同时验证空行不影响最小缩进，并保留本来顶格的代码。

- [ ] **Step 3: 运行并确认红灯**

```powershell
.venv\Scripts\python.exe -m unittest tests.test_site_structure -v
node tests\code-rendering.test.js
```

Expected: 品牌、H1、背景与 `dedentCode` 契约失败。

- [ ] **Step 4: 提交契约测试**

```powershell
git add tests/test_site_structure.py tests/code-rendering.test.js
git commit -m "test: define sitewide note polish contract"
```

### Task 2: 统一视觉与代码块缩进

**Files:**
- Modify: `docs/style.css`
- Modify: `docs/note.css`
- Modify: `docs/index.html`
- Modify: `docs/note.html`
- Modify: `docs/llm/ppo.html`
- Modify: `docs/llm/grpo.html`
- Modify: `docs/llm/concepts.html`
- Modify: `docs/code-rendering.js`

**Interfaces:**
- Consumes: `CodeRendering.decorateCodeBlock(block, highlighter)`。
- Produces: `CodeRendering.dedentCode(source): string`，并在高亮前应用。

- [ ] **Step 1: 实现 `dedentCode`**

按换行拆分文本，忽略空行计算最小前导空格或 tab 数，从每个非空行删除该宽度；无公共缩进时原样返回。`decorateCodeBlock` 在解析语言后、高亮前更新 `block.textContent`。

- [ ] **Step 2: 更新中性基础表面**

将 `--bg-soft` 和 `--accent-mist` 改为中性浅灰；品牌标识、表头、引用块、导航悬停不再使用粉色铺底。行内代码的莫兰迪粉背景保持不变，旧蓝色活动边框替换为莫兰迪粉透明色。

- [ ] **Step 3: 更新所有品牌入口与缓存版本**

把五个 HTML 入口的 `JZ` 改为 `Z`。首页本地资源版本升为 `20260810-3`，阅读器统一升为 `20260810-5`，同步修改结构测试期望。

- [ ] **Step 4: 运行测试**

```powershell
node tests\code-rendering.test.js
.venv\Scripts\python.exe -m unittest tests.test_site_structure.HomepageTests tests.test_site_structure.ViewerIntegrationTests -v
node --check docs\code-rendering.js
```

- [ ] **Step 5: 提交共享界面修改**

```powershell
git add docs/style.css docs/note.css docs/index.html docs/note.html docs/llm/ppo.html docs/llm/grpo.html docs/llm/concepts.html docs/code-rendering.js tests/test_site_structure.py
git commit -m "refactor: neutralize note surfaces and code indentation"
```

### Task 3: 并行整理 nano-vLLM 笔记

**Files:**
- Modify: `docs/infra/nano-vllm/*.md`
- Create if necessary: `docs/assets/infra/nano-vllm/*.drawio`
- Create if necessary: `docs/assets/infra/nano-vllm/*.svg`

**Interfaces:**
- Consumes: 已有统一阅读器与 Markdown 相对资源解析。
- Produces: 6 篇具有 H1、具体问题标题、行内代码和完整围栏的笔记。

- [ ] **Step 1: 审阅六篇原稿并保留技术主题**
- [ ] **Step 2: 修正 H1/H2/H3、大小写、空格、围栏和公共代码缩进**
- [ ] **Step 3: 在正文中为函数、类、变量、参数和枚举补行内代码**
- [ ] **Step 4: 将元叙述问题标题改为具体主题标题**
- [ ] **Step 5: 仅在执行链确实复杂时创建配套 draw.io/SVG 并嵌入**
- [ ] **Step 6: 运行 Markdown 围栏与全站结构测试并报告修改文件**

### Task 4: 并行整理 CUDA 笔记

**Files:**
- Modify: `docs/infra/cuda/cuda内存.md`
- Modify: `docs/infra/cuda/elementwise.md`
- Modify: `docs/infra/cuda/基础.md`
- Preserve empty: `docs/infra/cuda/点乘_softmax_norm.md`
- Create if necessary: `docs/assets/infra/cuda/*.drawio`
- Create if necessary: `docs/assets/infra/cuda/*.svg`

**Interfaces:**
- Consumes: 已有统一阅读器。
- Produces: 3 篇格式统一的非空 CUDA 笔记，空白笔记保持零内容。

- [ ] **Step 1: 审阅三篇非空原稿，确认代码与说明对应**
- [ ] **Step 2: 修正标题、大小写、围栏和公共代码缩进**
- [ ] **Step 3: 为 CUDA API、kernel、变量、类型和短表达式补行内代码**
- [ ] **Step 4: 仅为复杂内存流或 kernel 数据流增加 draw.io/SVG**
- [ ] **Step 5: 确认空白笔记未写入任何内容**
- [ ] **Step 6: 运行 Markdown 围栏与全站结构测试并报告修改文件**

### Task 5: 并行整理 LLM、RL 与 Concepts 笔记

**Files:**
- Modify: `docs/llm/rl/ppo.md`
- Modify: `docs/llm/rl/grpo.md`
- Modify: `docs/llm/concepts/kvcache.md`
- Preserve empty: `docs/llm/concepts/concepts.md`
- Preserve empty: `docs/llm/concepts/gae.md`
- Preserve empty: `docs/llm/concepts/index.md`
- Create if necessary: `docs/assets/llm/**/*.drawio`
- Create if necessary: `docs/assets/llm/**/*.svg`

**Interfaces:**
- Consumes: 已有数学、代码和 Markdown 渲染能力。
- Produces: PPO、GRPO、KV Cache 三篇格式统一的笔记，三个空白文件保持空白。

- [ ] **Step 1: 审阅三篇非空原稿，不改动公式语义**
- [ ] **Step 2: 将问题标题统一为具体主题名并修正标题层级**
- [ ] **Step 3: 为模型、函数、变量、张量、参数与短表达式补行内代码**
- [ ] **Step 4: 复用已有文字执行链；只有新增图能明显降低理解成本时才创建 draw.io/SVG**
- [ ] **Step 5: 确认三个空白文件未写入任何内容**
- [ ] **Step 6: 运行 Markdown 围栏与全站结构测试并报告修改文件**

### Task 6: 整理 vLLM 并创建 Greedy Sampling 流程图

**Files:**
- Modify: `docs/infra/vllm/sampling.md`
- Modify: `docs/infra/vllm/scheduler.md`
- Create: `docs/assets/infra/vllm/greedy-sampling-flow.drawio`
- Create: `docs/assets/infra/vllm/greedy-sampling-flow.svg`
- Modify: `tests/test_site_structure.py`

**Interfaces:**
- Consumes: `../../assets/infra/vllm/greedy-sampling-flow.svg` Markdown 相对路径。
- Produces: Greedy/Random/混合 Batch 决策流程图与两篇结构完整的 vLLM 笔记。

- [ ] **Step 1: 写失败的 vLLM 图形测试**

测试 `.drawio` 包含 `mxGraphModel`，SVG 的 `viewBox` 为 `0 0 1200 700`，且文本包含 `temperature`、`GREEDY`、`all_greedy`、`all_random`、`argmax`、`top-k / top-p`、`torch.where`。

- [ ] **Step 2: 创建可编辑图与发布 SVG**

图形按“参数归类 → Batch 分类 → Greedy/Random 分支 → 混合合并”组织，白底、`#b98991` 描边、`#8e5f68` 箭头、`#3f3638` 文字。

- [ ] **Step 3: 重写 Sampling 与 Scheduler 结构**

补 H1，修复残缺围栏，使用具体标题和行内代码；在 Greedy Sampling 章节嵌入 SVG。保持源码事实与原稿主题，不扩写无关版本细节。

- [ ] **Step 4: 运行 vLLM 和围栏测试**

```powershell
.venv\Scripts\python.exe -m unittest tests.test_site_structure.InfraNotePilotTests tests.test_site_structure.MarkdownCodeFenceTests -v
```

- [ ] **Step 5: 提交 vLLM 修改**

```powershell
git add docs/infra/vllm docs/assets/infra/vllm tests/test_site_structure.py
git commit -m "docs: polish vllm notes and sampling flow"
```

### Task 7: 整合、复核与浏览器验收

**Files:**
- Verify: all changed files

**Interfaces:**
- Consumes: Tasks 1–6 的整合结果。
- Produces: 可交付的全站统一阅读体验。

- [ ] **Step 1: 审查三个代理的差异与目录边界**
- [ ] **Step 2: 修复代理遗漏并统一措辞、链接与资源路径**
- [ ] **Step 3: 运行完整测试与语法检查**

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
```

- [ ] **Step 4: HTTP 验证首页、Sampling、SVG 与代表性目录页面均返回 200**
- [ ] **Step 5: 浏览器验证桌面三栏、移动抽屉、白色基础表面、`Z` 标识、代码顶格和 Sampling 图**
- [ ] **Step 6: 提交代理笔记整合并保持工作区干净**
