# Homepage and Notes Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current multi-section homepage with a white, Morandi-pink, single-column Projects/Notes/Plan interface and reorganize Markdown notes into expandable Infra and LLM hierarchies that open in the shared note viewer.

**Architecture:** Keep the site dependency-free and static. Use native `<details>/<summary>` elements for the homepage directory tree, a small reusable `note-paths.js` module for path validation and sibling-note collections, and the existing `note.html` Markdown viewer for all new links. Add Python and Node built-in tests so structure, paths, and viewer behavior can be checked without installing packages.

**Tech Stack:** HTML5, CSS custom properties, browser JavaScript, Python 3 `unittest`, Node.js `node:test`, GitHub Pages.

## Global Constraints

- The homepage content modules are exactly Projects, Notes, and Plan in a single column.
- Projects and Plan have no published entries in this change.
- The light theme background is `#FFFFFF`; approved Morandi accents are `#8E5F68`, `#B98991`, `#F4E8EA`, and `#FAF5F6`.
- Infra contains `cuda`, `nano-vllm`, and `vllm`; LLM contains `rl` and `concepts`.
- Homepage note navigation expands first-level directory, then second-level directory, then exposes Markdown links.
- Existing standalone `.html` notes remain compatible but are not linked from the homepage.
- `docs/code-agent/`, `docs/elementwise.md`, and `docs/learn.md` remain unchanged and unlisted.
- No runtime or test dependency may be added.

---

### Task 1: Reorganize Markdown Notes

**Files:**
- Create: `tests/test_site_structure.py`
- Move: `docs/cuda/` to `docs/infra/cuda/`
- Move: `docs/nano-vllm/` to `docs/infra/nano-vllm/`
- Move: `docs/vllm/` to `docs/infra/vllm/`
- Move: `docs/llm/ppo.md` and `docs/llm/grpo.md` to `docs/llm/rl/`
- Move: `docs/llm/concepts.md`, `docs/llm/gae.md`, `docs/llm/index.md`, and `docs/llm/kvcache.md` to `docs/llm/concepts/`

**Interfaces:**
- Consumes: The approved directory mapping in `docs/superpowers/specs/2026-08-10-homepage-notes-reorganization-design.md`.
- Produces: `EXPECTED_NOTE_PATHS: tuple[str, ...]` and filesystem topology used by later homepage and viewer tests.

- [ ] **Step 1: Write the failing topology test**

Create `tests/test_site_structure.py` with `unittest`, a repository root derived from `Path(__file__).resolve().parents[1]`, and the exact expected relative paths. Add tests that assert every expected Markdown path exists, old Infra roots do not exist, and moved LLM Markdown files no longer exist at `docs/llm/*.md`.

```python
EXPECTED_NOTE_PATHS = (
    "infra/cuda/cuda内存.md",
    "infra/cuda/elementwise.md",
    "infra/cuda/点乘_softmax_norm.md",
    "infra/cuda/基础.md",
    "infra/nano-vllm/kvcache_and_paged_attention.md",
    "infra/nano-vllm/llm_engine.md",
    "infra/nano-vllm/ModelRunner.md",
    "infra/nano-vllm/run_nano-vllm.md",
    "infra/nano-vllm/Scheduler.md",
    "infra/nano-vllm/struct.md",
    "infra/vllm/sampling.md",
    "infra/vllm/scheduler.md",
    "llm/rl/grpo.md",
    "llm/rl/ppo.md",
    "llm/concepts/concepts.md",
    "llm/concepts/gae.md",
    "llm/concepts/index.md",
    "llm/concepts/kvcache.md",
)
```

- [ ] **Step 2: Run the topology test and verify the expected failure**

Run: `.venv\Scripts\python.exe -m unittest tests.test_site_structure -v`

Expected: FAIL because `docs/infra/` and the new `docs/llm/rl/` and `docs/llm/concepts/` locations do not exist.

- [ ] **Step 3: Move the note directories and Markdown files**

Resolve and print every source and destination path before moving. Confirm each path begins with `E:/code/cakeman.github.io/docs/`. Create only `docs/infra/`, `docs/llm/rl/`, and `docs/llm/concepts/`, then use PowerShell `Move-Item -LiteralPath` for the exact directories and files listed above.

- [ ] **Step 4: Run the topology test and verify it passes**

Run: `.venv\Scripts\python.exe -m unittest tests.test_site_structure -v`

Expected: PASS with the exact 18 Markdown paths present and obsolete locations absent.

- [ ] **Step 5: Commit the directory reorganization**

```powershell
git add tests/test_site_structure.py docs/infra docs/llm
git commit -m "refactor: organize infra and llm notes"
```

---

### Task 2: Build the Single-Column Homepage

**Files:**
- Modify: `tests/test_site_structure.py`
- Modify: `docs/index.html`
- Modify: `docs/style.css`
- Modify: `docs/script.js`

**Interfaces:**
- Consumes: The new Markdown locations from Task 1.
- Produces: Native `.note-directory` disclosure elements, `.note-file-link` links with `note.html?path=...` URLs, empty `.project-list` and `.plan-list` containers, and approved light-theme CSS tokens.

- [ ] **Step 1: Add failing homepage structure and link tests**

Extend `tests/test_site_structure.py` with an `HTMLParser` subclass that collects section IDs, navigation targets, summary labels, CSS classes, and note-link `href` values. Add separate tests asserting:

```python
self.assertEqual(parser.section_ids, ["projects", "notes", "plan"])
self.assertEqual(parser.nav_targets, ["#projects", "#notes", "#plan"])
self.assertEqual(parser.top_level_summaries, ["Infra", "LLM"])
self.assertFalse(parser.project_entries)
self.assertFalse(parser.plan_entries)
```

For every `.note-file-link`, parse the `path` query parameter with `urllib.parse`, decode it, and assert `(DOCS_DIR / path).is_file()`. Add CSS assertions for `#FFFFFF`, `#8E5F68`, `#B98991`, `#F4E8EA`, and `#FAF5F6`, and assert the removed `hero`, `about`, and `contact` section IDs are absent.

- [ ] **Step 2: Run the homepage tests and verify the expected failure**

Run: `.venv\Scripts\python.exe -m unittest tests.test_site_structure -v`

Expected: FAIL because the current homepage still contains Hero, About, Contact, three project cards, old note links, and blue-green light-theme tokens.

- [ ] **Step 3: Replace the homepage markup with the approved structure**

Rewrite `docs/index.html` so the header contains the brand, the three navigation links, and the theme control. The main element contains only `section#projects`, `section#notes`, and `section#plan`. Use nested `<details class="note-directory">` and `<summary>` elements for Infra/LLM and their second-level directories. Add one `.note-file-link` per expected Markdown file using the shared viewer query form. Keep `.project-list` and `.plan-list` empty and show concise empty-state text outside those lists.

- [ ] **Step 4: Replace the homepage CSS with a single-column responsive system**

Rewrite `docs/style.css` around shared theme variables. Use the approved white/Morandi palette for `[data-theme="light"]`, a readable dark palette for the default theme, a narrow content width, divider-based sections, native disclosure markers, visible focus states, and mobile wrapping. Define `.plan-item` and `.plan-status` for future square completion indicators without adding plan entries.

- [ ] **Step 5: Simplify homepage JavaScript**

Update `docs/script.js` to retain theme persistence, header scroll state, current-year handling if present, and section-aware active navigation. Remove menu and card-grid behavior that no longer has matching markup. Do not add custom directory toggle code because native disclosure elements own that state.

- [ ] **Step 6: Run homepage and topology tests**

Run: `.venv\Scripts\python.exe -m unittest tests.test_site_structure -v`

Expected: PASS with exactly three homepage modules, all 18 links resolving, no project or plan entries, and all approved palette tokens present.

- [ ] **Step 7: Commit the homepage redesign**

```powershell
git add tests/test_site_structure.py docs/index.html docs/style.css docs/script.js
git commit -m "feat: simplify homepage navigation"
```

---

### Task 3: Update the Shared Note Viewer

**Files:**
- Create: `docs/note-paths.js`
- Create: `tests/note-paths.test.js`
- Modify: `docs/note.html`
- Modify: `docs/note.js`
- Modify: `docs/llm/ppo.html`
- Modify: `docs/llm/grpo.html`
- Modify: `docs/llm/concepts.html`
- Modify: `tests/test_site_structure.py`

**Interfaces:**
- Consumes: Markdown paths and homepage links from Tasks 1 and 2.
- Produces: `NotePaths.DEFAULT_NOTE_PATH: string`, `NotePaths.NOTE_COLLECTIONS: Record<string, NoteCollection>`, `NotePaths.sanitizeNotePath(rawPath): string | null`, `NotePaths.getCollectionForPath(path): NoteCollection | null`, and `NotePaths.createViewerHref(path, prefix): string`.

- [ ] **Step 1: Write failing path-module unit tests**

Create `tests/note-paths.test.js` with `node:test` and `node:assert/strict`. Require `../docs/note-paths.js` and assert:

```javascript
assert.equal(NotePaths.sanitizeNotePath("infra/vllm/scheduler.md"), "infra/vllm/scheduler.md");
assert.equal(NotePaths.sanitizeNotePath("../secret.md"), null);
assert.equal(NotePaths.sanitizeNotePath("https://example.com/a.md"), null);
assert.equal(NotePaths.sanitizeNotePath("infra/vllm/scheduler.txt"), null);
assert.deepEqual(
  NotePaths.getCollectionForPath("llm/rl/ppo.md").items.map((item) => item.path),
  ["llm/rl/ppo.md", "llm/rl/grpo.md"]
);
assert.equal(
  NotePaths.createViewerHref("infra/vllm/scheduler.md", ""),
  "note.html?path=infra%2Fvllm%2Fscheduler.md"
);
```

- [ ] **Step 2: Run the Node test and verify the expected failure**

Run: `node --test tests/note-paths.test.js`

Expected: FAIL because `docs/note-paths.js` does not exist.

- [ ] **Step 3: Implement the reusable path and collection module**

Create `docs/note-paths.js` as a browser-compatible IIFE that assigns the exact API to `globalThis.NotePaths` and also assigns `module.exports` when CommonJS is present. Define five collections keyed by `infra/cuda`, `infra/nano-vllm`, `infra/vllm`, `llm/rl`, and `llm/concepts`. Validation rejects traversal, absolute paths, URL schemes, backslashes, and non-Markdown extensions; invalid input returns `null`.

- [ ] **Step 4: Run the Node path tests and verify they pass**

Run: `node --test tests/note-paths.test.js`

Expected: PASS for valid paths, rejected paths, collection lookup, and encoded viewer links.

- [ ] **Step 5: Update viewer loading and sibling navigation**

Load `note-paths.js` immediately before `note.js` in `docs/note.html`. Replace the hard-coded RL catalog and fallback sanitization in `docs/note.js` with `globalThis.NotePaths`. Build sibling navigation from `getCollectionForPath(activePath)`, use encoded `note.html?path=...` links, and display an explicit "Invalid note path" error when sanitization returns `null`. Preserve fetch errors as explicit missing-note errors.

- [ ] **Step 6: Preserve standalone HTML compatibility**

Add `<script src="../note-paths.js"></script>` immediately before the existing `../note.js` script in `docs/llm/ppo.html`, `docs/llm/grpo.html`, and `docs/llm/concepts.html`. Update their `data-note-path` values to `llm/rl/ppo.md`, `llm/rl/grpo.md`, and `llm/concepts/concepts.md` respectively. Viewer link resolution uses `../note.html?path=...` and Markdown fetches use `../` when running from these one-level compatibility pages.

- [ ] **Step 7: Add static viewer integration checks**

Extend `tests/test_site_structure.py` to assert `note.html` loads `note-paths.js` before `note.js`, each standalone compatibility page loads both scripts in that order and uses its updated `data-note-path`, and no old `rl/ppo.md`, `rl/grpo.md`, or `rl/concepts.md` literal remains in `docs/note.js`, `docs/note.html`, or the compatibility-page body attributes.

- [ ] **Step 8: Run the complete automated suite**

Run: `.venv\Scripts\python.exe -m unittest tests.test_site_structure -v`

Run: `node --test tests/note-paths.test.js`

Expected: both suites PASS with no warnings or errors.

- [ ] **Step 9: Commit the viewer update**

```powershell
git add docs/note-paths.js docs/note.html docs/note.js docs/llm/ppo.html docs/llm/grpo.html docs/llm/concepts.html tests
git commit -m "feat: navigate reorganized markdown notes"
```

---

### Task 4: Browser Verification and Final Cleanup

**Files:**
- Modify only if verification exposes a failing acceptance criterion: `docs/index.html`, `docs/style.css`, `docs/script.js`, `docs/note.html`, `docs/note.js`, or `docs/note-paths.js`
- Modify with any corresponding regression first: `tests/test_site_structure.py` or `tests/note-paths.test.js`

**Interfaces:**
- Consumes: The completed homepage and note viewer at `http://127.0.0.1:8000/`.
- Produces: Browser evidence that native disclosures, note navigation, light theme, keyboard semantics, and responsive layout satisfy the design.

- [ ] **Step 1: Run all automated checks from a clean command invocation**

Run: `.venv\Scripts\python.exe -m unittest tests.test_site_structure -v`

Run: `node --test tests/note-paths.test.js`

Run: `git diff --check`

Expected: all tests PASS and `git diff --check` produces no output.

- [ ] **Step 2: Verify desktop homepage behavior in the browser**

Open `http://127.0.0.1:8000/`, switch to light mode, and confirm the white background, Morandi pink active navigation, single-column Projects/Notes/Plan sections, empty Projects and Plan, and collapsed Infra/LLM directories.

- [ ] **Step 3: Verify the complete expansion path**

Expand Infra, expand vllm, and confirm `sampling.md` and `scheduler.md` appear. Leave Infra open, expand LLM and rl, and confirm both first-level groups remain open. Click `ppo.md` and verify the URL contains `note.html?path=llm%2Frl%2Fppo.md`, the PPO Markdown renders, and the sibling list contains GRPO and PPO with PPO active.

- [ ] **Step 4: Verify mobile reflow and keyboard behavior**

At a viewport near 390 pixels wide, confirm there is no horizontal overflow and header navigation wraps without overlap. Use keyboard focus to open a first-level and second-level disclosure and follow a note link; verify focus indicators remain visible.

- [ ] **Step 5: Fix any observed failure through a regression-first cycle**

For each observed defect, add a focused failing Python or Node regression, run it to observe the expected failure, make the smallest production change, then rerun both automated suites and the affected browser flow.

- [ ] **Step 6: Commit verified cleanup only if files changed during browser verification**

```powershell
git add docs tests
git commit -m "fix: polish homepage note navigation"
```

- [ ] **Step 7: Record final repository state**

Run: `git status --short`

Run: `git log -4 --oneline`

Expected: only intentionally ignored visual-companion session files may remain untracked; all website, test, spec, and plan changes are committed.
