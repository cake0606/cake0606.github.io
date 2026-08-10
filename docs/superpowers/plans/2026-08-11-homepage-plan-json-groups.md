# Homepage Plan JSON Groups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the approved Infra and LLM plan groups on the homepage and make `docs/plan.json` the only file that must be edited for future plan-content changes.

**Architecture:** Keep the site build-free. `docs/plan.json` owns ordered plan data, `docs/index.html` exposes a loading-state mount, and `docs/script.js` validates the entire payload before creating semantic DOM nodes; invalid data produces one non-fatal error state. Python tests lock the static data and mount contract, while a Node VM test exercises browser-side loading, rendering, accessibility state, empty data, and failures.

**Tech Stack:** Static HTML5, CSS, browser JavaScript, JSON, Python `unittest`, Node.js `node:test`/`vm`; no new dependencies or build tooling.

## Global Constraints

- Publish directly from `docs`; do not introduce a package manager, bundler, framework, or third-party dependency.
- `docs/plan.json` is the sole source of Plan group titles, item text, order, and completion status.
- Keep the supplied copy exactly as approved, including lowercase titles and the spelling `continguous batching`.
- Render `Plan` as the existing `<h2>` and the `infra`/`llm` group titles as `<h3>` elements.
- Groups remain permanently expanded; do not add `<details>`, disclosure buttons, filters, progress summaries, or in-page editing.
- Completion is display-only. Visitors cannot toggle state; maintainers change only the JSON boolean.
- Validate the whole JSON payload. Any invalid root, group, or item produces `Plans are temporarily unavailable.` rather than a partial list.
- A valid empty root array produces `No plans published yet.`.
- Use `textContent` for JSON-controlled strings; do not interpolate plan content into `innerHTML`.
- Preserve the existing Projects, Notes, header, navigation, palette, and single-column layout.

## File Structure

- Create `docs/plan.json`: ordered Plan groups and completion state; this is the maintainer-facing content file.
- Create `tests/plan-runtime.test.js`: isolated runtime contract for fetching, validating, rendering, and failure handling.
- Modify `tests/test_site_structure.py`: replace the obsolete empty-Plan assertion with JSON and mount contracts.
- Modify `docs/index.html`: replace the empty list with a Plan mount and loading state; bump changed asset versions.
- Modify `docs/script.js`: add focused validation, DOM construction, state display, and asynchronous loading functions without disturbing navigation behavior.
- Modify `docs/style.css`: add Plan group/title spacing while reusing existing item and status-square rules.

---

### Task 1: Establish the Plan JSON data contract

**Files:**
- Create: `docs/plan.json`
- Modify: `tests/test_site_structure.py:1-55,134-180,343-375`

**Interfaces:**
- Consumes: the approved two-group copy from `docs/superpowers/specs/2026-08-11-homepage-plan-json-groups-design.md`.
- Produces: a JSON root of `Array<{title: string, items: Array<{text: string, completed: boolean}>}>` in `docs/plan.json`; Task 2 fetches this exact shape.

- [ ] **Step 1: Write the failing static data test**

Add `import json` with the standard-library imports. Define the exact expected content near the existing path constants:

```python
EXPECTED_PLAN_GROUPS = (
    (
        "infra",
        (
            "prefill与decode",
            "显存计算",
            "fa原理",
            "模型量化",
            "kvcache量化",
            "投机解码",
            "稀疏注意力与长上下文",
            "continguous batching",
            "chunked prefill",
            "cuda graph",
            "triton",
            "tp、pp、ep",
            "性能指标与分析工具",
        ),
    ),
    (
        "llm",
        (
            "mha、mqa、gqa",
            "rope、rmsnorm、swiglu",
            "dense与moe",
        ),
    ),
)
```

In `HomepageTests.setUpClass`, load the future file:

```python
cls.plan_data = json.loads((DOCS_DIR / "plan.json").read_text(encoding="utf-8"))
```

Replace `test_plan_publishes_no_entries` with the complete schema/content assertion:

```python
def test_plan_data_matches_the_approved_initial_content(self):
    self.assertIsInstance(self.plan_data, list)
    actual_groups = tuple(
        (group["title"], tuple(item["text"] for item in group["items"]))
        for group in self.plan_data
    )
    self.assertEqual(actual_groups, EXPECTED_PLAN_GROUPS)

    for group in self.plan_data:
        with self.subTest(group=group["title"]):
            self.assertEqual(set(group), {"title", "items"})
            self.assertIsInstance(group["title"], str)
            self.assertTrue(group["title"].strip())
            self.assertIsInstance(group["items"], list)
            for item in group["items"]:
                self.assertEqual(set(item), {"text", "completed"})
                self.assertIsInstance(item["text"], str)
                self.assertTrue(item["text"].strip())
                self.assertIs(type(item["completed"]), bool)
                self.assertFalse(item["completed"])
```

Remove the now-unused `HomepageParser.plan_entries` field and the `.plan-item` collection branch; runtime entries must not be hard-coded into HTML.

- [ ] **Step 2: Run the focused test and verify that it fails for the missing data file**

Run:

```powershell
python -m unittest tests.test_site_structure.HomepageTests.test_plan_data_matches_the_approved_initial_content -v
```

Expected: ERROR with `FileNotFoundError` for `docs/plan.json`.

- [ ] **Step 3: Create the approved JSON data**

Create `docs/plan.json` with exactly:

```json
[
  {
    "title": "infra",
    "items": [
      { "text": "prefill与decode", "completed": false },
      { "text": "显存计算", "completed": false },
      { "text": "fa原理", "completed": false },
      { "text": "模型量化", "completed": false },
      { "text": "kvcache量化", "completed": false },
      { "text": "投机解码", "completed": false },
      { "text": "稀疏注意力与长上下文", "completed": false },
      { "text": "continguous batching", "completed": false },
      { "text": "chunked prefill", "completed": false },
      { "text": "cuda graph", "completed": false },
      { "text": "triton", "completed": false },
      { "text": "tp、pp、ep", "completed": false },
      { "text": "性能指标与分析工具", "completed": false }
    ]
  },
  {
    "title": "llm",
    "items": [
      { "text": "mha、mqa、gqa", "completed": false },
      { "text": "rope、rmsnorm、swiglu", "completed": false },
      { "text": "dense与moe", "completed": false }
    ]
  }
]
```

- [ ] **Step 4: Run the focused test and full Python suite**

Run:

```powershell
python -m unittest tests.test_site_structure.HomepageTests.test_plan_data_matches_the_approved_initial_content -v
python -m unittest discover -s tests -p "test_*.py" -v
```

Expected: both commands PASS; the focused test confirms 2 groups, 16 ordered items, exact copy, exact keys, and 16 `false` booleans.

- [ ] **Step 5: Commit the data contract**

```powershell
git add -- docs/plan.json tests/test_site_structure.py
git commit -m "feat: add homepage plan data contract"
```

---

### Task 2: Render grouped Plan data with resilient states

**Files:**
- Create: `tests/plan-runtime.test.js`
- Modify: `tests/test_site_structure.py:134-190,343-385`
- Modify: `docs/index.html:17,118-128`
- Modify: `docs/script.js:1-44`
- Modify: `docs/style.css:331-359`

**Interfaces:**
- Consumes: `docs/plan.json` as `Array<{title: string, items: Array<{text: string, completed: boolean}>}>`.
- Produces: `validatePlanData(data) -> data`, `createPlanGroup(group) -> HTMLElement`, `renderPlanData(data) -> void`, `showPlanState(message) -> void`, and `loadPlan() -> Promise<void>` inside `docs/script.js`; DOM ids `plan-groups` and `plan-state`; CSS classes `plan-groups`, `plan-group`, `plan-group-title`, `plan-list`, `plan-item`, `plan-status`, and `is-complete`.

- [ ] **Step 1: Add the failing homepage mount contract**

Extend `HomepageParser.__init__`:

```python
self.plan_mounts = []
self.plan_states = []
```

Extend `HomepageParser.handle_starttag` after computing `attributes` and `classes`:

```python
if attributes.get("id") == "plan-groups":
    self.plan_mounts.append((tag, attributes))

if attributes.get("id") == "plan-state":
    self.plan_states.append((tag, attributes))
```

Add this test to `HomepageTests`:

```python
def test_plan_exposes_the_runtime_mount_and_state(self):
    self.assertEqual(len(self.parser.plan_mounts), 1)
    mount_tag, mount = self.parser.plan_mounts[0]
    self.assertEqual(mount_tag, "div")
    self.assertIn("plan-groups", mount["class"].split())
    self.assertEqual(mount["aria-live"], "polite")
    self.assertEqual(mount["aria-busy"], "true")

    self.assertEqual(len(self.parser.plan_states), 1)
    state_tag, state = self.parser.plan_states[0]
    self.assertEqual(state_tag, "p")
    self.assertIn("empty-state", state["class"].split())
    self.assertEqual(state["role"], "status")
```

- [ ] **Step 2: Add the failing browser-runtime tests**

Create `tests/plan-runtime.test.js`. The harness must execute the real homepage script while keeping existing navigation dependencies inert:

```javascript
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const source = fs.readFileSync(
  path.join(__dirname, "..", "docs", "script.js"),
  "utf8"
);

function createElement(tagName = "div") {
  const classes = new Set();
  const element = {
    tagName: tagName.toUpperCase(),
    children: [],
    attributes: {},
    hidden: false,
    textContent: "",
    classList: {
      add(...names) { names.forEach((name) => classes.add(name)); },
      contains(name) { return classes.has(name); },
      toggle(name, force) {
        const enabled = force === undefined ? !classes.has(name) : Boolean(force);
        if (enabled) classes.add(name);
        else classes.delete(name);
        return enabled;
      }
    },
    append(...nodes) { this.children.push(...nodes); },
    replaceChildren(...nodes) { this.children = [...nodes]; },
    setAttribute(name, value) { this.attributes[name] = String(value); },
    getAttribute(name) { return this.attributes[name]; },
    addEventListener() {}
  };

  Object.defineProperty(element, "className", {
    get() { return [...classes].join(" "); },
    set(value) {
      classes.clear();
      String(value).split(/\s+/).filter(Boolean).forEach((name) => classes.add(name));
    }
  });

  return element;
}

function createHarness(fetchImpl) {
  const header = createElement("header");
  const planGroups = createElement("div");
  const planState = createElement("p");
  const errors = [];
  const fetchCalls = [];

  const context = {
    console: { error(...args) { errors.push(args); } },
    document: {
      createElement,
      getElementById(id) {
        return id === "plan-groups" ? planGroups : id === "plan-state" ? planState : null;
      },
      querySelector(selector) { return selector === ".site-header" ? header : null; },
      querySelectorAll() { return []; }
    },
    fetch(url, options) {
      fetchCalls.push({ url, options });
      return fetchImpl(url, options);
    },
    window: {
      addEventListener() {},
      location: { hash: "" },
      scrollY: 0
    }
  };
  context.globalThis = context;
  vm.runInNewContext(source, context);

  return {
    errors,
    fetchCalls,
    header,
    planGroups,
    planState,
    async settle() {
      await new Promise((resolve) => setImmediate(resolve));
      await new Promise((resolve) => setImmediate(resolve));
    }
  };
}

test("loads grouped plans without caching and renders semantic status", async () => {
  const data = [
    {
      title: "infra",
      items: [
        { text: "prefill与decode", completed: false },
        { text: "cuda graph", completed: true }
      ]
    },
    { title: "llm", items: [{ text: "mha、mqa、gqa", completed: false }] }
  ];
  const harness = createHarness(async () => ({ ok: true, json: async () => data }));
  await harness.settle();

  assert.equal(harness.fetchCalls.length, 1);
  assert.equal(harness.fetchCalls[0].url, "plan.json");
  assert.equal(harness.fetchCalls[0].options.cache, "no-store");
  assert.equal(harness.planGroups.getAttribute("aria-busy"), "false");
  assert.equal(harness.planState.hidden, true);
  assert.equal(harness.planGroups.children.length, 2);

  const [infra, llm] = harness.planGroups.children;
  assert.equal(infra.tagName, "SECTION");
  assert.equal(infra.children[0].tagName, "H3");
  assert.equal(infra.children[0].textContent, "infra");
  assert.equal(llm.children[0].textContent, "llm");

  const items = infra.children[1].children;
  assert.equal(items[0].children[0].getAttribute("aria-label"), "未完成");
  assert.equal(items[0].children[1].textContent, "prefill与decode");
  assert.equal(items[1].classList.contains("is-complete"), true);
  assert.equal(items[1].children[0].getAttribute("aria-label"), "已完成");
});

test("shows the empty state for a valid empty plan", async () => {
  const harness = createHarness(async () => ({ ok: true, json: async () => [] }));
  await harness.settle();
  assert.equal(harness.planGroups.children.length, 0);
  assert.equal(harness.planGroups.getAttribute("aria-busy"), "false");
  assert.equal(harness.planState.hidden, false);
  assert.equal(harness.planState.textContent, "No plans published yet.");
});

test("shows one non-fatal error state for request, parse, HTTP, and schema failures", async (t) => {
  const cases = [
    ["request", async () => { throw new Error("offline"); }],
    ["HTTP", async () => ({ ok: false, status: 404, json: async () => [] })],
    ["parse", async () => ({ ok: true, json: async () => { throw new SyntaxError("bad json"); } })],
    ["schema", async () => ({ ok: true, json: async () => [{ title: "", items: [] }] })]
  ];

  for (const [name, fetchImpl] of cases) {
    await t.test(name, async () => {
      const harness = createHarness(fetchImpl);
      await harness.settle();
      assert.equal(harness.planGroups.children.length, 0);
      assert.equal(harness.planGroups.getAttribute("aria-busy"), "false");
      assert.equal(harness.planState.textContent, "Plans are temporarily unavailable.");
      assert.equal(harness.planState.hidden, false);
      assert.equal(harness.errors.length, 1);
    });
  }
});
```

- [ ] **Step 3: Run the new tests and verify that they fail for the missing mount and loader**

Run:

```powershell
python -m unittest tests.test_site_structure.HomepageTests.test_plan_exposes_the_runtime_mount_and_state -v
node --test tests/plan-runtime.test.js
```

Expected: the Python test FAILS because `plan_mounts` is empty; the Node tests FAIL because `fetch` is never called and no groups or error states are rendered.

- [ ] **Step 4: Replace the static empty list with the runtime mount**

Replace the Plan body after its heading in `docs/index.html` with:

```html
<div
  id="plan-groups"
  class="plan-groups"
  aria-live="polite"
  aria-busy="true"
></div>
<p id="plan-state" class="empty-state" role="status">Loading plans…</p>
```

Bump both changed homepage assets from `v=20260810-3` to `v=20260811-1`:

```html
<link rel="stylesheet" href="style.css?v=20260811-1">
<script src="script.js?v=20260811-1"></script>
```

- [ ] **Step 5: Implement full-payload validation and semantic rendering**

Add these functions to `docs/script.js` without changing the existing navigation functions:

```javascript
const planGroups = document.getElementById("plan-groups");
const planState = document.getElementById("plan-state");

function validatePlanData(data) {
  if (!Array.isArray(data)) {
    throw new TypeError("Plan data must be an array.");
  }

  data.forEach((group, groupIndex) => {
    if (!group || typeof group !== "object" || Array.isArray(group)) {
      throw new TypeError(`Plan group ${groupIndex} must be an object.`);
    }
    if (typeof group.title !== "string" || !group.title.trim()) {
      throw new TypeError(`Plan group ${groupIndex} must have a non-empty title.`);
    }
    if (!Array.isArray(group.items)) {
      throw new TypeError(`Plan group ${groupIndex} must have an items array.`);
    }

    group.items.forEach((item, itemIndex) => {
      if (!item || typeof item !== "object" || Array.isArray(item)) {
        throw new TypeError(`Plan item ${groupIndex}:${itemIndex} must be an object.`);
      }
      if (typeof item.text !== "string" || !item.text.trim()) {
        throw new TypeError(`Plan item ${groupIndex}:${itemIndex} must have non-empty text.`);
      }
      if (typeof item.completed !== "boolean") {
        throw new TypeError(`Plan item ${groupIndex}:${itemIndex} must have a boolean completed state.`);
      }
    });
  });

  return data;
}

function createPlanGroup(group) {
  const section = document.createElement("section");
  section.className = "plan-group";

  const title = document.createElement("h3");
  title.className = "plan-group-title";
  title.textContent = group.title;

  const list = document.createElement("ul");
  list.className = "plan-list";
  list.setAttribute("aria-label", `${group.title} plan`);

  group.items.forEach((item) => {
    const listItem = document.createElement("li");
    listItem.className = "plan-item";
    listItem.classList.toggle("is-complete", item.completed);

    const status = document.createElement("span");
    status.className = "plan-status";
    status.setAttribute("role", "img");
    status.setAttribute("aria-label", item.completed ? "已完成" : "未完成");

    const text = document.createElement("span");
    text.className = "plan-text";
    text.textContent = item.text;

    listItem.append(status, text);
    list.append(listItem);
  });

  section.append(title, list);
  return section;
}

function showPlanState(message) {
  planGroups.replaceChildren();
  planGroups.setAttribute("aria-busy", "false");
  planState.textContent = message;
  planState.hidden = false;
}

function renderPlanData(data) {
  const groups = data.map(createPlanGroup);
  planGroups.replaceChildren(...groups);
  planGroups.setAttribute("aria-busy", "false");

  if (groups.length === 0) {
    planState.textContent = "No plans published yet.";
    planState.hidden = false;
    return;
  }

  planState.hidden = true;
}

async function loadPlan() {
  if (!planGroups || !planState) return;

  try {
    const response = await fetch("plan.json", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Plan request failed with HTTP ${response.status}.`);
    }
    renderPlanData(validatePlanData(await response.json()));
  } catch (error) {
    console.error("Failed to load plan data:", error);
    showPlanState("Plans are temporarily unavailable.");
  }
}
```

Call `loadPlan();` once after the function definitions. Keep `updateHeaderState();` and `setActiveLink(...)` in place so a rejected Plan request cannot interrupt navigation initialization.

- [ ] **Step 6: Add the Plan group styles**

Insert before the existing `.plan-list` rule in `docs/style.css`:

```css
.plan-groups {
  display: grid;
  gap: 2rem;
}

.plan-group-title {
  margin: 0 0 0.35rem;
  color: var(--text);
  font-size: 1rem;
  line-height: 1.4;
}
```

Add a top divider to each group list while preserving the existing bottom dividers:

```css
.plan-list {
  display: grid;
  gap: 0;
  margin: 0;
  padding: 0;
  border-top: 1px solid var(--border);
  list-style: none;
}
```

Do not add a mobile media query: the existing single-column flow and wrapping already apply at narrow widths.

- [ ] **Step 7: Run the focused contracts and fix only contract failures**

Run:

```powershell
python -m unittest tests.test_site_structure.HomepageTests.test_plan_exposes_the_runtime_mount_and_state -v
python -m unittest tests.test_site_structure.HomepageTests.test_plan_data_matches_the_approved_initial_content -v
node --test tests/plan-runtime.test.js
```

Expected: PASS. The runtime suite records exactly one `plan.json` request with `{ cache: "no-store" }`, renders `<section>/<h3>/<ul>/<li>`, distinguishes the completed item, shows the valid-empty state, and handles all four failure classes without throwing out of initialization.

- [ ] **Step 8: Run all automated regression checks**

Run:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
node --test tests/*.test.js
git diff --check
```

Expected: all Python and Node tests PASS; `git diff --check` prints no errors.

- [ ] **Step 9: Verify the rendered page at desktop and narrow widths**

Start the static server from the repository root:

```powershell
$planServer = Start-Process -FilePath python -ArgumentList "-m", "http.server", "8000", "--directory", "docs" -WorkingDirectory (Get-Location) -WindowStyle Hidden -PassThru
```

Open `http://127.0.0.1:8000/` and verify:

1. Plan shows `infra` followed by 13 items, then `llm` followed by 3 items.
2. All 16 squares are visibly incomplete and aligned with wrapped text.
3. Neither group has a disclosure marker or click behavior.
4. At a viewport near 360 px wide, titles and long Chinese text wrap without horizontal overflow.
5. The browser console has no error and the `plan.json` request returns HTTP 200.

Stop the server:

```powershell
Stop-Process -Id $planServer.Id
```

- [ ] **Step 10: Commit the runtime, page, styles, and tests**

```powershell
git add -- docs/index.html docs/script.js docs/style.css tests/test_site_structure.py tests/plan-runtime.test.js
git commit -m "feat: render grouped homepage plans"
```

- [ ] **Step 11: Confirm the final tree is clean and the maintainer workflow works**

Run:

```powershell
git status --short
```

Expected: no output. Confirm that future content work needs only these JSON operations: append a group object, append an item object, edit `text`, reorder arrays, or change `completed` from `false` to `true`.
