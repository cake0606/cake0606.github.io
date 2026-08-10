from html.parser import HTMLParser
from pathlib import Path
import re
import unittest
from urllib.parse import parse_qs, urlparse
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"

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
    "llm/rl/ppo.md",
    "llm/rl/grpo.md",
    "llm/concepts/concepts.md",
    "llm/concepts/gae.md",
    "llm/concepts/index.md",
    "llm/concepts/kvcache.md",
)

OBSOLETE_INFRA_DIRECTORIES = (
    "cuda",
    "nano-vllm",
    "vllm",
)

MOVED_LLM_MARKDOWN = (
    "ppo.md",
    "grpo.md",
    "concepts.md",
    "gae.md",
    "index.md",
    "kvcache.md",
)

CODE_FENCE_PATTERN = re.compile(r"^\s*(?:>\s*)?(```|~~~)([^\s`]*)\s*$")


class HomepageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.section_ids = []
        self.nav_targets = []
        self.note_links = []
        self.project_entries = []
        self.plan_entries = []
        self.top_level_summaries = []
        self.second_level_summaries = []
        self.local_assets = []
        self.theme_toggles = []
        self._details_levels = []
        self._summary_level = None
        self._summary_text = []

    @staticmethod
    def _attributes(attrs):
        return dict(attrs)

    @staticmethod
    def _classes(attributes):
        return set(attributes.get("class", "").split())

    def handle_starttag(self, tag, attrs):
        attributes = self._attributes(attrs)
        classes = self._classes(attributes)

        if tag == "section" and attributes.get("id"):
            self.section_ids.append(attributes["id"])

        if tag == "a" and "nav-link" in classes:
            self.nav_targets.append(attributes.get("href"))

        if tag == "a" and "note-file-link" in classes:
            self.note_links.append(attributes.get("href"))

        if tag == "link" and attributes.get("rel") == "stylesheet":
            href = attributes.get("href", "")
            if not href.startswith(("http://", "https://")):
                self.local_assets.append(href)

        if tag == "script" and attributes.get("src"):
            self.local_assets.append(attributes["src"])

        if {"project-item", "project-card"} & classes:
            self.project_entries.append(tag)

        if "plan-item" in classes:
            self.plan_entries.append(tag)

        if "theme-toggle" in classes:
            self.theme_toggles.append(tag)

        if tag == "details":
            self._details_levels.append(attributes.get("data-level"))

        if tag == "summary":
            self._summary_level = self._details_levels[-1] if self._details_levels else None
            self._summary_text = []

    def handle_data(self, data):
        if self._summary_level is not None:
            self._summary_text.append(data)

    def handle_endtag(self, tag):
        if tag == "summary" and self._summary_level is not None:
            label = " ".join("".join(self._summary_text).split())
            if self._summary_level == "1":
                self.top_level_summaries.append(label)
            elif self._summary_level == "2":
                self.second_level_summaries.append(label)
            self._summary_level = None
            self._summary_text = []

        if tag == "details" and self._details_levels:
            self._details_levels.pop()


class ViewerDocumentParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.script_sources = []
        self.local_assets = []
        self.stylesheet_sources = []
        self.theme_toggles = []
        self.note_path = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "script" and attributes.get("src"):
            self.script_sources.append(urlparse(attributes["src"]).path)
            if not attributes["src"].startswith(("http://", "https://")):
                self.local_assets.append(attributes["src"])
        if tag == "link" and attributes.get("rel") == "stylesheet":
            href = attributes.get("href", "")
            self.stylesheet_sources.append(urlparse(href).path)
            if not href.startswith(("http://", "https://")):
                self.local_assets.append(href)
        if tag == "body":
            self.note_path = attributes.get("data-note-path")
        if "theme-toggle" in set(attributes.get("class", "").split()):
            self.theme_toggles.append(tag)


class NoteTopologyTests(unittest.TestCase):
    def test_every_organized_note_exists(self):
        """Removing or misplacing an approved note must break the site catalog."""
        for relative_path in EXPECTED_NOTE_PATHS:
            with self.subTest(relative_path=relative_path):
                self.assertTrue((DOCS_DIR / relative_path).is_file())

    def test_obsolete_infra_directories_are_gone(self):
        """Leaving duplicate Infra roots would make the hierarchy ambiguous."""
        for relative_path in OBSOLETE_INFRA_DIRECTORIES:
            with self.subTest(relative_path=relative_path):
                self.assertFalse((DOCS_DIR / relative_path).exists())

    def test_llm_markdown_is_not_duplicated_at_the_collection_root(self):
        """Leaving moved LLM notes at the old root would expose two sources."""
        for filename in MOVED_LLM_MARKDOWN:
            with self.subTest(filename=filename):
                self.assertFalse((DOCS_DIR / "llm" / filename).exists())


class MarkdownCodeFenceTests(unittest.TestCase):
    @staticmethod
    def markdown_files():
        for root in (DOCS_DIR / "infra", DOCS_DIR / "llm"):
            yield from root.rglob("*.md")

    def test_every_opening_fence_has_a_descriptive_language(self):
        """Unlabeled or abbreviated fences cannot produce reliable UI labels."""
        unlabeled_fences = []
        short_alias_fences = []
        unclosed_fences = []

        for path in self.markdown_files():
            open_marker = None
            open_line = None
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                match = CODE_FENCE_PATTERN.match(line)
                if not match:
                    continue

                marker, language = match.groups()
                if open_marker is None:
                    open_marker = marker
                    open_line = line_number
                    if not language:
                        unlabeled_fences.append((path.relative_to(DOCS_DIR), line_number))
                    elif language.lower() in {"py", "cu"}:
                        short_alias_fences.append(
                            (path.relative_to(DOCS_DIR), line_number, language.lower())
                        )
                elif marker == open_marker and not language:
                    open_marker = None
                    open_line = None

            if open_marker is not None:
                unclosed_fences.append((path.relative_to(DOCS_DIR), open_line))

        self.assertEqual(unlabeled_fences, [])
        self.assertEqual(short_alias_fences, [])
        self.assertEqual(unclosed_fences, [])


class HomepageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = HomepageParser()
        cls.parser.feed((DOCS_DIR / "index.html").read_text(encoding="utf-8"))
        cls.styles = (DOCS_DIR / "style.css").read_text(encoding="utf-8").upper()
        cls.theme_sources = "\n".join(
            (DOCS_DIR / path).read_text(encoding="utf-8").upper()
            for path in ("style.css", "note.css", "script.js", "note.js")
        )

    def test_homepage_contains_only_the_approved_modules(self):
        """Restoring an old Hero, About, or Contact module must fail."""
        self.assertEqual(self.parser.section_ids, ["projects", "notes", "plan"])

    def test_primary_navigation_targets_the_three_modules(self):
        """A missing or stale navigation link must fail."""
        self.assertEqual(self.parser.nav_targets, ["#projects", "#notes", "#plan"])

    def test_projects_publishes_no_entries(self):
        """Accidentally republishing old work must fail."""
        self.assertIn("projects", self.parser.section_ids)
        self.assertFalse(self.parser.project_entries)

    def test_plan_publishes_no_entries(self):
        """Omitting the empty Plan module or adding sample plans must fail."""
        self.assertIn("plan", self.parser.section_ids)
        self.assertFalse(self.parser.plan_entries)

    def test_note_tree_exposes_the_approved_directory_levels(self):
        """Flattening or omitting either directory level must fail."""
        self.assertEqual(self.parser.top_level_summaries, ["Infra", "LLM"])
        self.assertEqual(
            self.parser.second_level_summaries,
            ["CUDA", "nano-vLLM", "vLLM", "RL", "Concepts"],
        )

    def test_every_organized_note_is_linked_once_and_resolves(self):
        """A missing, duplicate, or broken Markdown link must fail."""
        resolved_paths = []
        for href in self.parser.note_links:
            query = parse_qs(urlparse(href).query)
            self.assertIn("path", query)
            path = query["path"][0]
            resolved_paths.append(path)
            self.assertTrue((DOCS_DIR / path).is_file(), path)

        self.assertCountEqual(resolved_paths, EXPECTED_NOTE_PATHS)
        self.assertEqual(len(resolved_paths), len(set(resolved_paths)))

    def test_light_theme_uses_the_approved_white_and_morandi_palette(self):
        """Replacing the approved light palette with the old teal palette must fail."""
        for color in ("#FFFFFF", "#8E5F68", "#B98991", "#F4E8EA", "#FAF5F6"):
            with self.subTest(color=color):
                self.assertIn(color, self.styles)

    def test_site_is_light_only(self):
        """Theme controls or dark-mode state would violate the single-theme design."""
        self.assertFalse(self.parser.theme_toggles)
        for forbidden in (
            "DATA-THEME",
            "SITE-THEME",
            "PREFERS-COLOR-SCHEME",
            "COLOR-SCHEME: DARK",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, self.theme_sources)

    def test_local_assets_are_versioned_to_prevent_stale_ui(self):
        """A cached pre-redesign stylesheet must not be mixed with the new HTML."""
        self.assertTrue(self.parser.local_assets)
        for asset in self.parser.local_assets:
            with self.subTest(asset=asset):
                self.assertEqual(parse_qs(urlparse(asset).query).get("v"), ["20260810-2"], asset)


class ViewerIntegrationTests(unittest.TestCase):
    @staticmethod
    def parse_document(relative_path):
        parser = ViewerDocumentParser()
        parser.feed((DOCS_DIR / relative_path).read_text(encoding="utf-8"))
        return parser

    def test_root_viewer_loads_path_module_before_viewer_logic(self):
        """Loading note.js before its path API would break every homepage note link."""
        parser = self.parse_document("note.html")
        self.assertIn("note-paths.js", parser.script_sources)
        self.assertIn("note.js", parser.script_sources)
        self.assertLess(
            parser.script_sources.index("note-paths.js"),
            parser.script_sources.index("note.js"),
        )

    def test_viewers_have_no_theme_toggle(self):
        """Every note entry point must use the same fixed light site shell."""
        for page in ("note.html", "llm/ppo.html", "llm/grpo.html", "llm/concepts.html"):
            with self.subTest(page=page):
                self.assertFalse(self.parse_document(page).theme_toggles)

    def test_compatibility_pages_load_existing_organized_notes(self):
        """A stale embedded path or missing path module would break legacy URLs."""
        compatibility_pages = {
            "llm/ppo.html": "llm/rl/ppo.md",
            "llm/grpo.html": "llm/rl/grpo.md",
            "llm/concepts.html": "llm/concepts/concepts.md",
        }

        for page, expected_note_path in compatibility_pages.items():
            with self.subTest(page=page):
                parser = self.parse_document(page)
                self.assertIn("../note-paths.js", parser.script_sources)
                self.assertIn("../note.js", parser.script_sources)
                self.assertLess(
                    parser.script_sources.index("../note-paths.js"),
                    parser.script_sources.index("../note.js"),
                )
                self.assertEqual(parser.note_path, expected_note_path)
                self.assertTrue((DOCS_DIR / parser.note_path).is_file())

    def test_viewer_assets_are_versioned_to_prevent_stale_navigation(self):
        """Viewer HTML, scripts, and styles must update as one deployable unit."""
        for page in ("note.html", "llm/ppo.html", "llm/grpo.html", "llm/concepts.html"):
            with self.subTest(page=page):
                parser = self.parse_document(page)
                self.assertTrue(parser.local_assets)
                for asset in parser.local_assets:
                    self.assertEqual(
                        parse_qs(urlparse(asset).query).get("v"),
                        ["20260810-3"],
                        asset,
                    )

    def test_viewers_load_full_highlighter_code_module_and_catppuccin_theme(self):
        """Every viewer must load the browser build and local Mocha theme in order."""
        viewer_expectations = {
            "note.html": ("code-rendering.js", "vendor/catppuccin-mocha.css"),
            "llm/ppo.html": ("../code-rendering.js", "../vendor/catppuccin-mocha.css"),
            "llm/grpo.html": ("../code-rendering.js", "../vendor/catppuccin-mocha.css"),
            "llm/concepts.html": ("../code-rendering.js", "../vendor/catppuccin-mocha.css"),
        }
        full_highlighter = "/ajax/libs/highlight.js/11.11.1/highlight.min.js"

        for page, (code_module, theme_stylesheet) in viewer_expectations.items():
            with self.subTest(page=page):
                parser = self.parse_document(page)
                self.assertIn(full_highlighter, parser.script_sources)
                self.assertIn(code_module, parser.script_sources)
                self.assertLess(
                    parser.script_sources.index(code_module),
                    parser.script_sources.index("note.js" if page == "note.html" else "../note.js"),
                )
                self.assertIn(theme_stylesheet, parser.stylesheet_sources)

        self.assertTrue((DOCS_DIR / "vendor" / "catppuccin-mocha.css").is_file())

    def test_note_styles_define_language_label_and_morandi_inline_code(self):
        """Code chrome must expose its language and avoid the retired blue-gray chip."""
        styles = (DOCS_DIR / "note.css").read_text(encoding="utf-8").upper()
        self.assertIn("PRE::BEFORE", styles)
        self.assertIn("CONTENT: ATTR(DATA-LANGUAGE)", styles)
        for color in ("#F4E8EA", "#E8DDDF", "#8E5F68"):
            self.assertIn(color, styles)

    def test_viewers_do_not_load_commonjs_highlight_language_modules(self):
        """Node-oriented Highlight.js modules throw `module is not defined` in browsers."""
        for page in ("note.html", "llm/ppo.html", "llm/grpo.html", "llm/concepts.html"):
            with self.subTest(page=page):
                parser = self.parse_document(page)
                self.assertFalse(
                    any("/lib/languages/" in source for source in parser.script_sources),
                    parser.script_sources,
                )


class InfraNotePilotTests(unittest.TestCase):
    NOTE_PATH = DOCS_DIR / "infra" / "nano-vllm" / "kvcache_and_paged_attention.md"
    DRAWIO_PATH = DOCS_DIR / "assets" / "infra" / "nano-vllm" / "kv-cache-dataflow.drawio"
    SVG_PATH = DOCS_DIR / "assets" / "infra" / "nano-vllm" / "kv-cache-dataflow.svg"

    def test_kv_cache_diagram_has_editable_source_and_accessible_svg(self):
        """The published diagram must stay editable, scalable, and understandable."""
        self.assertTrue(self.DRAWIO_PATH.is_file())
        self.assertTrue(self.SVG_PATH.is_file())

        drawio_root = ET.parse(self.DRAWIO_PATH).getroot()
        self.assertEqual(drawio_root.tag, "mxfile")
        self.assertIsNotNone(drawio_root.find("./diagram/mxGraphModel/root"))

        svg_root = ET.parse(self.SVG_PATH).getroot()
        self.assertEqual(svg_root.attrib.get("viewBox"), "0 0 1200 560")
        self.assertEqual(svg_root.attrib.get("role"), "img")
        self.assertEqual(svg_root.attrib.get("aria-labelledby"), "title desc")
        svg_text = " ".join(svg_root.itertext())
        for label in (
            "逻辑层",
            "映射层",
            "物理层",
            "Sequence",
            "BlockManager",
            "block_table",
            "prepare_prefill / prepare_decode",
            "slot_mapping",
            "store_kvcache",
            "KV Cache Tensor",
        ):
            self.assertIn(label, svg_text)

    def test_kv_cache_note_has_the_pilot_structure(self):
        """The pilot note must present one coherent, reproducible mapping walkthrough."""
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
            "[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]",
            "slot = block_id * block_size + offset",
            "block_size = 4",
            "block_table = [7, 2]",
            "slots = [28, 29, 30, 31, 8, 9]",
            "can_allocate",
            "allocate",
            "may_append",
            "hash_blocks",
            "deallocate",
            "ref_count",
            "完整块",
        ):
            self.assertIn(required, markdown)


if __name__ == "__main__":
    unittest.main()
