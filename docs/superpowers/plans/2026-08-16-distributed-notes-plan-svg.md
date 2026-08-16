# Distributed Notes, Plan, and SVG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the complete Distributed note collection, mark DP and DDP complete in the Infra plan, and remove the confirmed overlap from the DDP bucket diagram.

**Architecture:** Keep the existing static-site architecture: `docs/index.html`, `docs/note-paths.js`, and `docs/plan.json` remain the explicit navigation and data sources. Extend the existing Python and Node test contracts before changing content, then make a targeted SVG layout change with machine-checkable geometry metadata and rendered visual verification.

**Tech Stack:** Static HTML/CSS/JavaScript, Markdown, JSON, SVG, Python `unittest`, Node.js built-in test runner, Sharp for local SVG rasterization.

## Global Constraints

- Keep `docs/` directly publishable without adding a build step or runtime directory scan.
- Homepage Infra directory order must be CUDA, Distributed, nano-vLLM, vLLM.
- Distributed note order must be `基础.md`, `DP-DDP.md`, `deepspeed.md`, `tp-ep-pp.md`.
- Rename `deepspped.md` to `deepspeed.md` and keep its content empty.
- Add `DP、DDP` as completed and `DeepSpeed` as incomplete after `triton`; keep `tp、pp、ep` incomplete.
- Do not change `ring-all-reduce-bucket.svg` content.
- Keep the DDP SVG at `viewBox="0 0 1400 860"` with its existing title, description, color system, and semantics.
- Use forward slashes in every published note path.

---

### Task 1: Publish the Distributed note collection

**Files:**
- Modify: `tests/test_site_structure.py`
- Modify: `tests/note-paths.test.js`
- Modify: `docs/index.html`
- Modify: `docs/note-paths.js`
- Rename: `docs/infra/distributed/deepspped.md` to `docs/infra/distributed/deepspeed.md`

**Interfaces:**
- Consumes: the existing `HomepageParser`, `EXPECTED_NOTE_PATHS`, and `NotePaths.NOTE_COLLECTIONS` contracts.
- Produces: an `infra/distributed` note collection whose ordered item paths are `基础.md`, `DP-DDP.md`, `deepspeed.md`, and `tp-ep-pp.md`.

- [ ] **Step 1: Add failing Python structure expectations**

Add these paths to `EXPECTED_NOTE_PATHS` immediately after the CUDA paths:

```python
    "infra/distributed/基础.md",
    "infra/distributed/DP-DDP.md",
    "infra/distributed/deepspeed.md",
    "infra/distributed/tp-ep-pp.md",
```

Add `"infra/distributed/deepspeed.md"` to `EMPTY_NOTE_PATHS`, and change the expected second-level summaries to:

```python
[
    "CUDA",
    "Distributed",
    "nano-vLLM",
    "vLLM",
    "RL",
    "Concepts",
]
```

- [ ] **Step 2: Add failing Node collection and asset-resolution expectations**

Extend `returns the sibling collection for an organized note` with:

```javascript
  assert.deepEqual(
    NotePaths.getCollectionForPath("infra/distributed/DP-DDP.md").items.map(
      (item) => item.path
    ),
    [
      "infra/distributed/基础.md",
      "infra/distributed/DP-DDP.md",
      "infra/distributed/deepspeed.md",
      "infra/distributed/tp-ep-pp.md"
    ]
  );
```

Extend `resolves relative note assets from the Markdown source directory` with:

```javascript
  assert.equal(
    NotePaths.resolveNoteAssetHref(
      "infra/distributed/DP-DDP.md",
      "../../assets/infra/distributed/ddp-bucket-overlap.svg",
      ""
    ),
    "assets/infra/distributed/ddp-bucket-overlap.svg"
  );
```

- [ ] **Step 3: Run focused tests and verify they fail**

Run:

```powershell
python -m unittest tests.test_site_structure.HomepageTests.test_note_tree_exposes_the_approved_directory_levels tests.test_site_structure.HomepageTests.test_every_organized_note_is_linked_once_and_resolves
node --test tests/note-paths.test.js
```

Expected: Python fails because Distributed is absent and `deepspeed.md` does not exist; Node fails because `infra/distributed` is not registered.

- [ ] **Step 4: Rename the typo and add the homepage directory**

Run the exact tracked rename:

```powershell
git mv -- docs/infra/distributed/deepspped.md docs/infra/distributed/deepspeed.md
```

Insert this `<details>` block after CUDA and before nano-vLLM in `docs/index.html`:

```html
            <details class="note-directory note-directory-child" data-level="2">
              <summary>Distributed</summary>
              <ul class="note-file-list">
                <li><a class="note-file-link" href="note.html?path=infra/distributed/基础.md">基础.md</a></li>
                <li><a class="note-file-link" href="note.html?path=infra/distributed/DP-DDP.md">DP-DDP.md</a></li>
                <li><a class="note-file-link" href="note.html?path=infra/distributed/deepspeed.md">deepspeed.md</a></li>
                <li><a class="note-file-link" href="note.html?path=infra/distributed/tp-ep-pp.md">tp-ep-pp.md</a></li>
              </ul>
            </details>
```

- [ ] **Step 5: Register the reader sibling collection**

Insert this collection after `infra/cuda` in `docs/note-paths.js`:

```javascript
    "infra/distributed": {
      title: "Distributed",
      items: [
        { title: "Distributed Basics", path: "infra/distributed/基础.md" },
        { title: "DP & DDP", path: "infra/distributed/DP-DDP.md" },
        { title: "DeepSpeed", path: "infra/distributed/deepspeed.md" },
        { title: "TP, EP & PP", path: "infra/distributed/tp-ep-pp.md" }
      ]
    },
```

- [ ] **Step 6: Run focused tests and verify they pass**

Run:

```powershell
python -m unittest tests.test_site_structure.HomepageTests.test_note_tree_exposes_the_approved_directory_levels tests.test_site_structure.HomepageTests.test_every_organized_note_is_linked_once_and_resolves tests.test_site_structure.MarkdownCodeFenceTests
node --test tests/note-paths.test.js
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit the note collection**

```powershell
git add -- docs/index.html docs/note-paths.js docs/infra/distributed tests/test_site_structure.py tests/note-paths.test.js
git commit -m "feat: publish distributed notes"
```

---

### Task 2: Update Infra plan completion states

**Files:**
- Modify: `tests/test_site_structure.py`
- Modify: `docs/plan.json`

**Interfaces:**
- Consumes: the existing grouped Plan JSON schema `{title: string, items: {text: string, completed: boolean}[]}`.
- Produces: exactly one completed item, `("infra", "DP、DDP")`, while `DeepSpeed` and `tp、pp、ep` remain incomplete.

- [ ] **Step 1: Add explicit Plan completion expectations**

Add the two texts to `EXPECTED_PLAN_GROUPS` after `triton`:

```python
            "DP、DDP",
            "DeepSpeed",
```

Define this constant after `EXPECTED_PLAN_GROUPS`:

```python
EXPECTED_COMPLETED_PLAN_ITEMS = {("infra", "DP、DDP")}
```

Replace the unconditional `self.assertFalse(item["completed"])` check in `test_plan_data_matches_the_approved_initial_content` with this assertion after schema validation:

```python
        actual_completed = {
            (group["title"], item["text"])
            for group in self.plan_data
            for item in group["items"]
            if item["completed"]
        }
        self.assertEqual(actual_completed, EXPECTED_COMPLETED_PLAN_ITEMS)
```

- [ ] **Step 2: Run the Plan structure test and verify it fails**

Run:

```powershell
python -m unittest tests.test_site_structure.HomepageTests.test_plan_data_matches_the_approved_initial_content
```

Expected: FAIL because `DP、DDP` and `DeepSpeed` are not yet in `docs/plan.json`.

- [ ] **Step 3: Add the two Plan entries**

Insert after the existing `triton` item in `docs/plan.json`:

```json
      { "text": "DP、DDP", "completed": true },
      { "text": "DeepSpeed", "completed": false },
```

Leave the following `tp、pp、ep` item unchanged with `completed: false`.

- [ ] **Step 4: Run Plan tests and verify they pass**

Run:

```powershell
python -m unittest tests.test_site_structure.HomepageTests.test_plan_data_matches_the_approved_initial_content
node --test tests/plan-runtime.test.js
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the Plan update**

```powershell
git add -- docs/plan.json tests/test_site_structure.py
git commit -m "feat: update distributed learning plan"
```

---

### Task 3: Make the DDP SVG overlap regression-testable and fix it

**Files:**
- Modify: `tests/test_site_structure.py`
- Modify: `docs/assets/infra/distributed/ddp-bucket-overlap.svg`
- Verify unchanged: `docs/assets/infra/distributed/ring-all-reduce-bucket.svg`

**Interfaces:**
- Consumes: XML/SVG parsing through `xml.etree.ElementTree` and existing `svg_node_bounds` rectangle support.
- Produces: accessible SVGs with valid local `url(#id)` references and DDP geometry IDs `hook-layer-1`, `gradient-bucket-1`, and `overlap-note`.

- [ ] **Step 1: Add Distributed SVG paths and rectangle-overlap helper**

Add these constants to `InfraNotePilotTests`:

```python
    DDP_NOTE_PATH = DOCS_DIR / "infra" / "distributed" / "DP-DDP.md"
    DDP_BUCKET_SVG_PATH = (
        DOCS_DIR / "assets" / "infra" / "distributed" / "ddp-bucket-overlap.svg"
    )
    RING_BUCKET_SVG_PATH = (
        DOCS_DIR
        / "assets"
        / "infra"
        / "distributed"
        / "ring-all-reduce-bucket.svg"
    )
```

Add this module-level helper after `collinear_overlap_length`:

```python
def bounds_overlap(first, second):
    first_left, first_top, first_right, first_bottom = first
    second_left, second_top, second_right, second_bottom = second
    return (
        max(first_left, second_left) < min(first_right, second_right)
        and max(first_top, second_top) < min(first_bottom, second_bottom)
    )
```

- [ ] **Step 2: Add failing SVG accessibility and overlap tests**

Add these methods to `InfraNotePilotTests`:

```python
    def test_distributed_note_references_accessible_svgs(self):
        markdown = self.DDP_NOTE_PATH.read_text(encoding="utf-8")
        expected_svgs = (
            (self.DDP_BUCKET_SVG_PATH, "0 0 1400 860", "ddp-bucket-overlap.svg"),
            (self.RING_BUCKET_SVG_PATH, "0 0 1400 820", "ring-all-reduce-bucket.svg"),
        )
        for svg_path, view_box, filename in expected_svgs:
            with self.subTest(svg=filename):
                self.assertTrue(svg_path.is_file())
                self.assertIn(f"../../assets/infra/distributed/{filename}", markdown)
                svg_root = ET.parse(svg_path).getroot()
                self.assertEqual(svg_root.attrib.get("viewBox"), view_box)
                self.assertEqual(svg_root.attrib.get("role"), "img")
                self.assertEqual(svg_root.attrib.get("aria-labelledby"), "title desc")
                self.assertIsNotNone(svg_root.find(f"{SVG_NAMESPACE}title"))
                self.assertIsNotNone(svg_root.find(f"{SVG_NAMESPACE}desc"))

                ids = {
                    element.attrib["id"]
                    for element in svg_root.iter()
                    if "id" in element.attrib
                }
                references = {
                    match.group(1)
                    for element in svg_root.iter()
                    for value in element.attrib.values()
                    for match in re.finditer(r"url\(#([^\)]+)\)", value)
                }
                self.assertLessEqual(references, ids)

    def test_ddp_overlap_note_clears_hook_and_bucket_nodes(self):
        svg_root = ET.parse(self.DDP_BUCKET_SVG_PATH).getroot()
        rectangles = {
            element.attrib["id"]: element
            for element in svg_root.iter(f"{SVG_NAMESPACE}rect")
            if "id" in element.attrib
        }
        required = {"hook-layer-1", "gradient-bucket-1", "overlap-note"}
        self.assertLessEqual(required, set(rectangles))

        overlap_bounds = svg_node_bounds(rectangles["overlap-note"])
        for node_id in ("hook-layer-1", "gradient-bucket-1"):
            self.assertFalse(
                bounds_overlap(overlap_bounds, svg_node_bounds(rectangles[node_id])),
                f"overlap note intersects {node_id}",
            )

        svg_text = " ".join(svg_root.itertext())
        self.assertIn(
            "并行：Bucket 0 All-Reduce 与 Layer 1 Backward 重叠",
            svg_text,
        )
```

- [ ] **Step 3: Run the SVG tests and verify the overlap test fails**

Run:

```powershell
python -m unittest tests.test_site_structure.InfraNotePilotTests.test_distributed_note_references_accessible_svgs tests.test_site_structure.InfraNotePilotTests.test_ddp_overlap_note_clears_hook_and_bucket_nodes
```

Expected: the accessibility test passes; the geometry test fails because the three regression IDs and replacement overlap note do not exist yet.

- [ ] **Step 4: Replace the colliding DDP callout with a dedicated left-side note**

In `ddp-bucket-overlap.svg`:

1. Add the style:

```svg
      .overlap-note { fill: #edf3f7; stroke: #7893a6; stroke-width: 1.5; stroke-dasharray: 6 5; }
```

2. Add `id="hook-layer-1"` to the rectangle at `x="1004" y="626"`.
3. Add `id="gradient-bucket-1"` to the Bucket 1 rectangle at `x="624" y="680"`.
4. Delete the final dashed path `M976 624 V660 H1138 V626` and its colliding text.
5. Insert this independent note before the Bucket 1 group:

```svg
  <rect id="overlap-note" class="overlap-note" x="44" y="638" width="540" height="50" rx="12"/>
  <text class="small" x="314" y="668" text-anchor="middle">并行：Bucket 0 All-Reduce 与 Layer 1 Backward 重叠</text>
```

- [ ] **Step 5: Run SVG tests and verify they pass**

Run:

```powershell
python -m unittest tests.test_site_structure.InfraNotePilotTests.test_distributed_note_references_accessible_svgs tests.test_site_structure.InfraNotePilotTests.test_ddp_overlap_note_clears_hook_and_bucket_nodes
```

Expected: both tests pass.

- [ ] **Step 6: Rasterize and visually inspect both SVGs**

Use the bundled Node and Sharp runtime to render the original viewBox sizes into the thread visualization directory. Confirm the DDP note does not intersect the hook or Bucket 1 and the Ring diagram remains unchanged and unclipped.

```powershell
$env:NODE_PATH='C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules'
$reviewDir='C:\Users\Administrator\.codex\visualizations\2026\08\16\01a0084a-77ce-7480-b3bc-1b7564f82212\distributed-final'
New-Item -ItemType Directory -Force -Path $reviewDir | Out-Null
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' -e "const sharp=require('sharp'); Promise.all([sharp('docs/assets/infra/distributed/ddp-bucket-overlap.svg').png().toFile('C:/Users/Administrator/.codex/visualizations/2026/08/16/01a0084a-77ce-7480-b3bc-1b7564f82212/distributed-final/ddp-bucket-overlap.png'),sharp('docs/assets/infra/distributed/ring-all-reduce-bucket.svg').png().toFile('C:/Users/Administrator/.codex/visualizations/2026/08/16/01a0084a-77ce-7480-b3bc-1b7564f82212/distributed-final/ring-all-reduce-bucket.png')]).then(console.log)"
```

The review PNGs remain outside the repository and must not be added to Git.

- [ ] **Step 7: Commit the SVG correction**

```powershell
git add -- docs/assets/infra/distributed/ddp-bucket-overlap.svg tests/test_site_structure.py
git commit -m "fix: separate DDP overlap annotation"
```

---

### Task 4: Run full verification

**Files:**
- Verify: all files changed by Tasks 1–3

**Interfaces:**
- Consumes: the final static site, note registry, Plan data, and SVG assets.
- Produces: evidence that all repository tests pass and no stale typo path remains.

- [ ] **Step 1: Run all Python tests**

```powershell
python -m unittest discover -s tests -p "test*.py"
```

Expected: PASS with zero failures and zero errors.

- [ ] **Step 2: Run all Node tests**

```powershell
node --test tests/code-rendering.test.js tests/note-paths.test.js tests/note-runtime.test.js tests/plan-runtime.test.js
```

Expected: PASS with zero failed tests.

- [ ] **Step 3: Check repository consistency**

Run:

```powershell
git diff --check
git status --short
rg -n "deepspped|infra/distributed|DP、DDP|DeepSpeed" docs tests
```

Expected: `git diff --check` emits nothing; `deepspped` appears only in historical design/plan text that documents the rename, while every live navigation path uses `deepspeed.md`; only the implementation plan remains uncommitted if it was not included in an earlier documentation commit.

- [ ] **Step 4: Review the final diff**

```powershell
git diff HEAD~3 -- docs tests
```

Expected: the diff contains only the approved Distributed navigation, Plan status, typo rename, DDP SVG layout correction, and their tests.
