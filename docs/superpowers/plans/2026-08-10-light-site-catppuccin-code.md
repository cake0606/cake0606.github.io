# Light-only site and Catppuccin code rendering implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove website theme switching, keep the approved light UI, and render every fenced code block with Catppuccin Mocha highlighting plus a visible language label.

**Architecture:** Keep site-level colors in `style.css` as a single light palette. Isolate language alias resolution in a small UMD module that can be unit tested in Node, while `note.js` applies the resolved metadata to rendered DOM blocks and invokes the full Highlight.js browser build. Store the Catppuccin theme locally and normalize Markdown fence identifiers without changing prose.

**Tech Stack:** Static HTML/CSS, vanilla JavaScript, Marked, Highlight.js 11.11.1 browser bundle, Catppuccin Mocha, Python `unittest`, Node test runner.

## Global Constraints

- The website has one global presentation: white and Morandi-pink light mode.
- Fenced code always uses Catppuccin Mocha and does not follow a site theme.
- Existing generic and compatibility note URLs remain valid.
- Markdown prose and technical examples remain unchanged; only fence identifiers may change.
- Inline code keeps Markdown backticks and receives the Morandi pink treatment.
- Do not add copy buttons, line numbers, or a code toolbar.
- Increment local static asset version parameters to `20260810-2`.

---

### Task 1: Remove site theme switching

**Files:**
- Modify: `tests/test_site_structure.py`
- Modify: `docs/index.html`
- Modify: `docs/note.html`
- Modify: `docs/llm/ppo.html`
- Modify: `docs/llm/grpo.html`
- Modify: `docs/llm/concepts.html`
- Modify: `docs/style.css`
- Modify: `docs/note.css`
- Modify: `docs/script.js`
- Modify: `docs/note.js`

**Interfaces:**
- Consumes: existing static page markup and header behavior.
- Produces: HTML with no `.theme-toggle`; one `:root` light palette; scripts with no theme persistence API.

- [ ] **Step 1: Write failing light-only tests**

Extend the HTML parsers to collect elements with class `theme-toggle`. Add tests over the homepage and all viewer documents that assert no toggle exists. Add source assertions that `style.css`, `script.js`, and `note.js` contain none of `data-theme`, `site-theme`, `prefers-color-scheme`, or `color-scheme: dark`, while `style.css` still contains `#FFFFFF`, `#8E5F68`, `#B98991`, `#F4E8EA`, and `#FAF5F6`.

```python
def test_site_is_light_only(self):
    self.assertFalse(self.parser.theme_toggles)
    for forbidden in ("DATA-THEME", "SITE-THEME", "PREFERS-COLOR-SCHEME", "COLOR-SCHEME: DARK"):
        self.assertNotIn(forbidden, self.combined_theme_sources)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `.venv\Scripts\python.exe -m unittest tests.test_site_structure.HomepageTests.test_site_is_light_only tests.test_site_structure.ViewerIntegrationTests.test_viewers_have_no_theme_toggle -v`

Expected: FAIL because theme buttons, dark variables, and theme JavaScript still exist.

- [ ] **Step 3: Implement the light-only shell**

Remove the theme buttons from all five HTML entry points. Promote the current `body[data-theme="light"]` variables to `:root`, set `color-scheme: light`, delete the dark variables and conditional selectors, and retain the existing white/Morandi values. Remove theme persistence and toggle listeners from both scripts while preserving header, note navigation, TOC, and responsive panel behavior. Promote the light inline-code and panel styles in `note.css` to their unconditional selectors.

- [ ] **Step 4: Run the complete structure suite and syntax checks**

Run:

```powershell
.venv\Scripts\python.exe -m unittest tests.test_site_structure -v
node --check docs/script.js
node --check docs/note.js
```

Expected: all tests and syntax checks PASS.

- [ ] **Step 5: Commit**

```powershell
git add tests/test_site_structure.py docs/index.html docs/note.html docs/llm/ppo.html docs/llm/grpo.html docs/llm/concepts.html docs/style.css docs/note.css docs/script.js docs/note.js
git commit -m "refactor: keep site in light mode"
```

### Task 2: Add testable language metadata and Catppuccin highlighting

**Files:**
- Create: `docs/code-rendering.js`
- Create: `docs/vendor/catppuccin-mocha.css`
- Create: `tests/code-rendering.test.js`
- Modify: `tests/test_site_structure.py`
- Modify: `docs/note.js`
- Modify: `docs/note.css`
- Modify: `docs/note.html`
- Modify: `docs/llm/ppo.html`
- Modify: `docs/llm/grpo.html`
- Modify: `docs/llm/concepts.html`

**Interfaces:**
- Produces: `globalThis.CodeRendering.resolveLanguage(classNames: string[]): { sourceLanguage: string, highlightLanguage: string | null, label: string }`.
- Produces: `globalThis.CodeRendering.decorateCodeBlock(block: HTMLElement, highlighter?: object): void`.
- Consumes: Marked output shaped as `pre > code.language-*` and optional global `hljs`.

- [ ] **Step 1: Write failing Node tests for language resolution**

Create assertions for `python`, `py`, `cuda`, `cu`, `bash`, `text`, and an unknown identifier. Require CUDA to highlight as `cpp` but label as `CUDA`, text to skip highlighting, and unknown identifiers to remain readable with a title-cased label.

```javascript
assert.deepEqual(resolveLanguage(["language-cuda"]), {
  sourceLanguage: "cuda",
  highlightLanguage: "cpp",
  label: "CUDA"
});
```

- [ ] **Step 2: Run the Node test and verify RED**

Run: `node tests/code-rendering.test.js`

Expected: FAIL because `docs/code-rendering.js` does not exist.

- [ ] **Step 3: Implement the UMD language module**

Define immutable aliases and labels. `decorateCodeBlock` must set `pre.dataset.language`, add `has-language-label`, replace unsupported aliases with the resolved `language-*` class before highlighting, call `highlightElement` only for a known highlight language, and add `hljs` without throwing when no highlighter is supplied.

- [ ] **Step 4: Verify the pure module GREEN**

Run: `node tests/code-rendering.test.js`

Expected: all language-resolution and decoration tests PASS.

- [ ] **Step 5: Write failing viewer integration tests**

Assert every viewer loads, in order, the full cdnjs browser bundle, `code-rendering.js?v=20260810-2`, and `note.js?v=20260810-2`. Assert every viewer links `vendor/catppuccin-mocha.css?v=20260810-2`. Assert `note.css` contains a `pre::before` language-label rule and the inline colors `#F4E8EA`, `#E8DDDF`, and `#8E5F68`.

- [ ] **Step 6: Run integration tests and verify RED**

Run: `.venv\Scripts\python.exe -m unittest tests.test_site_structure.ViewerIntegrationTests -v`

Expected: FAIL because the old core Highlight.js URL is present and the new module/theme/label contract is absent.

- [ ] **Step 7: Integrate the full browser build and local theme**

Replace the npm `lib/highlight.min.js` URL with `https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/highlight.min.js`. Save the official Catppuccin Mocha Highlight.js CSS as `docs/vendor/catppuccin-mocha.css`. Load the theme after the base page stylesheet and load `code-rendering.js` before `note.js` in every viewer.

Update `highlightCodeBlocks()` to call `CodeRendering.decorateCodeBlock(block, hljs)` for every rendered block. Add the language label with `pre::before { content: attr(data-language); }`, reserve top padding, keep horizontal scrolling, and replace inline code's blue-gray colors with the approved Morandi values. Remove the duplicate hand-maintained Highlight.js token palette from `note.css` so the vendored theme is authoritative.

- [ ] **Step 8: Run integration, Node, and syntax tests**

Run:

```powershell
.venv\Scripts\python.exe -m unittest tests.test_site_structure -v
node tests/code-rendering.test.js
node tests/note-paths.test.js
node --check docs/code-rendering.js
node --check docs/note.js
```

Expected: all tests PASS.

- [ ] **Step 9: Commit**

```powershell
git add docs/code-rendering.js docs/vendor/catppuccin-mocha.css tests/code-rendering.test.js tests/test_site_structure.py docs/note.js docs/note.css docs/note.html docs/llm/ppo.html docs/llm/grpo.html docs/llm/concepts.html
git commit -m "feat: render code with Catppuccin Mocha"
```

### Task 3: Normalize every Markdown code fence

**Files:**
- Modify: `tests/test_site_structure.py`
- Modify: Markdown files under `docs/infra/` and `docs/llm/` that contain short or missing fence identifiers.

**Interfaces:**
- Consumes: `CodeRendering.resolveLanguage` aliases from Task 2.
- Produces: every opening fence has an explicit identifier; CUDA fences use `cuda`, Python fences use `python`, non-code diagrams use `text`.

- [ ] **Step 1: Write a failing Markdown fence audit**

Add a helper that scans each Markdown file line by line, tracks opening and closing backtick or tilde fences, and records an error when an opening fence has no identifier. Also reject opening identifiers `py` and `cu` so aliases do not remain in stored notes.

```python
self.assertEqual(unlabeled_fences, [])
self.assertEqual(short_alias_fences, [])
```

- [ ] **Step 2: Run the audit and verify RED**

Run: `.venv\Scripts\python.exe -m unittest tests.test_site_structure.MarkdownCodeFenceTests -v`

Expected: FAIL for the two unlabeled blocks plus existing `py` and `cu` openings.

- [ ] **Step 3: Normalize fence identifiers only**

Change opening `cu` identifiers to `cuda`, `py` to `python`, and the two unlabeled diagram/formula blocks to `text`. Do not change closing fences, prose, indentation, triple-quoted Python strings, or code bodies.

- [ ] **Step 4: Run all tests**

Run:

```powershell
.venv\Scripts\python.exe -m unittest tests.test_site_structure -v
node tests/code-rendering.test.js
node tests/note-paths.test.js
```

Expected: the fence audit and all prior suites PASS.

- [ ] **Step 5: Commit**

```powershell
git add tests/test_site_structure.py docs/infra docs/llm
git commit -m "docs: label fenced code languages"
```

### Task 4: Browser and release verification

**Files:**
- Verify: `docs/index.html`
- Verify: `docs/note.html`
- Verify: representative notes in `docs/infra/` and `docs/llm/`

**Interfaces:**
- Consumes: completed light shell, language module, Catppuccin theme, and normalized Markdown.
- Produces: verified local UI with no required code changes unless a new failing regression test is added first.

- [ ] **Step 1: Run fresh automated verification**

```powershell
.venv\Scripts\python.exe -m unittest tests.test_site_structure -v
node tests/code-rendering.test.js
node tests/note-paths.test.js
node --check docs/script.js
node --check docs/note-paths.js
node --check docs/code-rendering.js
node --check docs/note.js
git diff --check
```

Expected: zero failures and zero syntax or whitespace errors.

- [ ] **Step 2: Verify representative browser behavior**

At `http://127.0.0.1:8000/`, confirm there is no theme button and the body stays white. Open representative Python, CUDA, Bash, and Text notes through the homepage hierarchy. For each, confirm the language label, Catppuccin token spans, dark Mocha block, pink inline code, readable overflow, and zero new console errors.

- [ ] **Step 3: Verify responsive layout**

Set the browser viewport to 390 by 844 pixels. Confirm `document.documentElement.scrollWidth <= document.documentElement.clientWidth`, language labels do not overlap the first code line, and code scrolls inside its block. Reset the viewport afterward.

- [ ] **Step 4: Verify repository state**

Run: `git status --short` and `git log -4 --oneline`.

Expected: clean working tree with the three implementation commits visible.
