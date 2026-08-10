# vLLM Sampling Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate Greedy/Random sampling and multi-sequence generation into one vLLM Sampling note, remove the old Scheduler note, and keep the unified diagram, navigation, and tests synchronized.

**Architecture:** `docs/infra/vllm/sampling.md` becomes the single canonical note with two second-level topics: “Greedy 与 Random” and “多序列生成”. The existing flow graph is retained but renamed and relabeled for its broader Greedy/Random scope; navigation and tests are updated so no repository surface points to the removed Scheduler note.

**Tech Stack:** Markdown, static HTML, JavaScript/CommonJS, SVG, Draw.io XML, Python `unittest`, Node.js test runner.

## Global Constraints

- Work directly on the existing `main` branch.
- Preserve the existing flowchart geometry and connector IDs.
- Keep Draw.io and SVG node/edge IDs synchronized.
- Do not add strategy comparison, key conclusions, references, or Speculative Decoding sections.
- Delete `docs/infra/vllm/scheduler.md` and every navigation entry for it.
- Keep Top-k and Top-p under Random candidate filtering rather than presenting them as peer sampling modes.

---

### Task 1: Lock the consolidated note contract in tests

**Files:**
- Modify: `tests/test_site_structure.py:53`
- Modify: `tests/test_site_structure.py:612`
- Modify: `tests/test_site_structure.py:687`
- Modify: `tests/test_site_structure.py:841`
- Modify: `tests/note-paths.test.js:6`

**Interfaces:**
- Consumes: the approved article hierarchy and asset name from the design spec.
- Produces: executable expectations for the canonical note path, removed Scheduler path, renamed diagram assets, required headings, and merged Parallel/Beam content.

- [ ] **Step 1: Update Python path and diagram constants**

Remove `"infra/vllm/scheduler.md"` from `EXPECTED_NOTE_PATHS`, add a constant for the removed path, and point the diagram constants at the new filenames:

```python
VLLM_SAMPLING_NOTE_PATH = DOCS_DIR / "infra" / "vllm" / "sampling.md"
VLLM_REMOVED_SCHEDULER_NOTE_PATH = DOCS_DIR / "infra" / "vllm" / "scheduler.md"
VLLM_SAMPLING_DRAWIO_PATH = (
    DOCS_DIR / "assets" / "infra" / "vllm" / "greedy-random-sampling-flow.drawio"
)
VLLM_SAMPLING_SVG_PATH = (
    DOCS_DIR / "assets" / "infra" / "vllm" / "greedy-random-sampling-flow.svg"
)
```

- [ ] **Step 2: Replace the Sampling note assertion with the new structure**

Assert the exact second-level headings, their order, the new image reference, and the merged multi-sequence terms:

```python
second_level_headings = re.findall(r"^## (.+)$", markdown, flags=re.MULTILINE)
self.assertEqual(second_level_headings, ["Greedy 与 Random", "多序列生成"])
self.assertIn(
    "![vLLM Greedy 与 Random Sampling 流程](../../assets/infra/vllm/greedy-random-sampling-flow.svg?v=20260811-2)",
    markdown,
)
for section in (
    "### 参数归一化与类型判定",
    "### Batch 聚合与执行路径",
    "### Logits 处理顺序",
    "### Random 候选过滤",
    "### Parallel Sampling",
    "### Beam Search",
):
    self.assertIn(section, markdown)
for identifier in (
    "`SamplingType.GREEDY`",
    "`all_greedy`",
    "`all_random`",
    "`torch.where`",
    "`ParentRequest`",
    "`BeamSearchInstance`",
):
    self.assertIn(identifier, markdown)
self.assertNotIn("## 关键结论", markdown)
self.assertNotIn("## 参考资料", markdown)
self.assertNotIn("## 两种策略对比", markdown)
self.assertNotIn("Speculative Decoding", markdown)
self.assertFalse(self.VLLM_REMOVED_SCHEDULER_NOTE_PATH.exists())
```

- [ ] **Step 3: Rename the diagram test helpers and check the broader title**

Rename the `test_greedy_sampling_*` methods and `_greedy_svg_geometry()` helper to `test_greedy_random_sampling_*` and `_greedy_random_svg_geometry()`. Add these title assertions without changing geometry expectations:

```python
self.assertEqual(svg_root.find(f"{SVG_NAMESPACE}title").text, "vLLM Greedy 与 Random Sampling 流程")
self.assertEqual(drawio_root.find("./diagram").attrib.get("name"), "Greedy and Random Sampling Flow")
```

- [ ] **Step 4: Update Node note-path tests to use the remaining vLLM note**

Replace Scheduler examples throughout the file with Sampling equivalents, including the approved path and the backslash, non-Markdown, and doubled-separator rejection cases. Make collection and link assertions canonical:

```javascript
assert.deepEqual(
  NotePaths.getCollectionForPath("infra/vllm/sampling.md").items.map((item) => item.path),
  ["infra/vllm/sampling.md"]
);
assert.equal(
  NotePaths.createViewerHref("infra/vllm/sampling.md", ""),
  "note.html?path=infra%2Fvllm%2Fsampling.md"
);
```

- [ ] **Step 5: Run the focused tests and verify they fail for the expected missing changes**

Run:

```powershell
python -m unittest tests.test_site_structure.InfraNotePilotTests -v
node --test tests/note-paths.test.js
```

Expected: FAIL because the new asset files, new headings, removed Scheduler note, and single-item vLLM navigation do not exist yet.

### Task 2: Consolidate the Markdown content and navigation

**Files:**
- Modify: `docs/infra/vllm/sampling.md:1`
- Delete: `docs/infra/vllm/scheduler.md`
- Modify: `docs/note-paths.js:32`
- Modify: `docs/index.html:84`

**Interfaces:**
- Consumes: the test contract from Task 1 and existing content from both vLLM notes.
- Produces: one canonical Sampling note and one-item vLLM navigation collection.

- [ ] **Step 1: Rewrite the Sampling opening and place the diagram after the concepts**

Start the body with `## Greedy 与 Random`. Explain that Greedy selects `argmax`, Random samples from a probability distribution, Temperature adjusts distribution sharpness, and Top-k/Top-p/Min-p filter Random candidates. Embed the new `greedy-random-sampling-flow.svg?v=20260811-2` reference immediately after this explanation.

- [ ] **Step 2: Reorganize the existing single-token execution content**

Move the existing code and explanations under these headings, preserving their technical details:

```markdown
### 参数归一化与类型判定
### Batch 聚合与执行路径
#### All Greedy
#### All Random
#### 混合 Batch
### Logits 处理顺序
### Random 候选过滤
#### Top-k
#### Top-p
```

State explicitly that Top-k/Top-p do not affect a Greedy result: pure Greedy returns before those processors, and mixed batches select `greedy_sampled` for Greedy rows after vectorized Random work.

- [ ] **Step 3: Merge Parallel Sampling and Beam Search**

Append `## 多序列生成`, introduce the shared fan-out/fan-in idea, then migrate the existing Scheduler content under:

```markdown
### Parallel Sampling
#### ParentRequest 的职责
#### 执行流程
### Beam Search
#### BeamSearchInstance
#### 每轮扩展流程
```

Keep child request aggregation, stable completion indexes, cumulative beam scores, and per-round beam pruning. Do not copy the comparison table, key-conclusion list, or references section.

- [ ] **Step 4: Remove the old note and navigation entries**

Delete `docs/infra/vllm/scheduler.md`, remove its item from `docs/note-paths.js`, and remove its `<li>` from `docs/index.html`. Leave Sampling as the only item in the `infra/vllm` collection.

- [ ] **Step 5: Run content and navigation tests**

Run:

```powershell
python -m unittest tests.test_site_structure.InfraNotePilotTests.test_sampling_note_covers_greedy_random_and_multi_sequence_generation -v
node --test tests/note-paths.test.js
```

Expected: the Node test passes; the Markdown structure assertions pass except any assertions that still depend on the renamed diagram files.

### Task 3: Rename and relabel the unified diagram

**Files:**
- Rename: `docs/assets/infra/vllm/greedy-sampling-flow.drawio` to `docs/assets/infra/vllm/greedy-random-sampling-flow.drawio`
- Rename: `docs/assets/infra/vllm/greedy-sampling-flow.svg` to `docs/assets/infra/vllm/greedy-random-sampling-flow.svg`
- Test: `tests/test_site_structure.py`

**Interfaces:**
- Consumes: the new asset paths and title asserted in Task 1.
- Produces: editable and published diagram assets with unchanged graph topology and updated Greedy/Random semantics.

- [ ] **Step 1: Rename both diagram assets**

Use exact source and destination paths so both editable and published variants move together.

- [ ] **Step 2: Update Draw.io metadata and visible title**

Set the `<diagram name>` to `Greedy and Random Sampling Flow` and replace the visible title text `vLLM Greedy Sampling` with `vLLM Greedy 与 Random Sampling`. Do not alter node IDs, edge IDs, source/target pairs, or connector geometry.

- [ ] **Step 3: Update SVG accessibility metadata and visible title**

Set the SVG `<title>` to `vLLM Greedy 与 Random Sampling 流程`, update `<desc>` to describe both Greedy and Random branches, and replace the visible heading with `vLLM Greedy 与 Random Sampling`. Keep `viewBox="0 0 1200 700"`, node bounds, and path data unchanged.

- [ ] **Step 4: Run diagram topology and connector tests**

Run:

```powershell
python -m unittest tests.test_site_structure.InfraNotePilotTests -v
```

Expected: PASS, including editable-source, topology, node clearance, edge overlap, lane-label clearance, and visible-arrow-shaft checks.

### Task 4: Full regression and browser verification

**Files:**
- Verify: `docs/infra/vllm/sampling.md`
- Verify: `docs/note-paths.js`
- Verify: `docs/index.html`
- Verify: `tests/test_site_structure.py`
- Verify: `tests/note-paths.test.js`

**Interfaces:**
- Consumes: the consolidated note, renamed diagram, removed note, updated navigation, and passing focused tests.
- Produces: repository-wide and visual evidence that the change is complete.

- [ ] **Step 1: Run the complete Python suite**

Run:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

Expected: all Python tests pass.

- [ ] **Step 2: Run the complete Node suite and syntax checks**

Run:

```powershell
node --test tests/*.test.js
node --check docs/note-paths.js
node --check docs/note.js
```

Expected: all Node tests pass and both JavaScript files parse without errors.

- [ ] **Step 3: Check for stale references and malformed diffs**

Run:

```powershell
rg -n "infra/vllm/scheduler\.md|greedy-sampling-flow|vLLM Greedy Sampling 流程" docs/index.html docs/note-paths.js docs/infra docs/assets tests
git diff --check
```

Expected: `rg` returns no stale production/test references, and `git diff --check` reports no whitespace errors.

- [ ] **Step 4: Verify the rendered note in the local browser**

Open `http://127.0.0.1:8000/note.html?path=infra/vllm/sampling.md`, reload it, and confirm:

- the navigation contains Sampling but not Scheduler;
- the table of contents has exactly two second-level topics;
- the concept explanation appears before the unified diagram;
- the diagram loads with the broader Greedy/Random title;
- Parallel Sampling and Beam Search render under “多序列生成”.

- [ ] **Step 5: Commit the completed implementation**

Run:

```powershell
git add -- docs/infra/vllm/sampling.md docs/note-paths.js docs/index.html docs/assets/infra/vllm tests/test_site_structure.py tests/note-paths.test.js
git add -u -- docs/infra/vllm/scheduler.md docs/assets/infra/vllm/greedy-sampling-flow.drawio docs/assets/infra/vllm/greedy-sampling-flow.svg
git commit -m "docs: consolidate vllm sampling strategies"
```
