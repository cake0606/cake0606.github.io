from html.parser import HTMLParser
from pathlib import Path
import unittest
from urllib.parse import parse_qs, urlparse


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
        self.note_path = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "script" and attributes.get("src"):
            self.script_sources.append(urlparse(attributes["src"]).path)
            if not attributes["src"].startswith(("http://", "https://")):
                self.local_assets.append(attributes["src"])
        if tag == "link" and attributes.get("rel") == "stylesheet":
            href = attributes.get("href", "")
            if not href.startswith(("http://", "https://")):
                self.local_assets.append(href)
        if tag == "body":
            self.note_path = attributes.get("data-note-path")


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


class HomepageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = HomepageParser()
        cls.parser.feed((DOCS_DIR / "index.html").read_text(encoding="utf-8"))
        cls.styles = (DOCS_DIR / "style.css").read_text(encoding="utf-8").upper()

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

    def test_local_assets_are_versioned_to_prevent_stale_ui(self):
        """A cached pre-redesign stylesheet must not be mixed with the new HTML."""
        self.assertTrue(self.parser.local_assets)
        for asset in self.parser.local_assets:
            with self.subTest(asset=asset):
                self.assertTrue(urlparse(asset).query, asset)


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
                    self.assertTrue(urlparse(asset).query, asset)

    def test_viewers_do_not_load_commonjs_highlight_language_modules(self):
        """Node-oriented Highlight.js modules throw `module is not defined` in browsers."""
        for page in ("note.html", "llm/ppo.html", "llm/grpo.html", "llm/concepts.html"):
            with self.subTest(page=page):
                parser = self.parse_document(page)
                self.assertFalse(
                    any("/lib/languages/" in source for source in parser.script_sources),
                    parser.script_sources,
                )


if __name__ == "__main__":
    unittest.main()
