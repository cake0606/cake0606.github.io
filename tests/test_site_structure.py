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

EMPTY_NOTE_PATHS = (
    "infra/cuda/点乘_softmax_norm.md",
    "llm/concepts/concepts.md",
    "llm/concepts/gae.md",
    "llm/concepts/index.md",
)

CODE_FENCE_PATTERN = re.compile(r"^\s*(?:>\s*)?(```|~~~)([^\s`]*)\s*$")
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
        return top < y1 < bottom and max(min(x1, x2), left) < min(
            max(x1, x2), right
        )
    if x1 == x2:
        return left < x1 < right and max(min(y1, y2), top) < min(
            max(y1, y2), bottom
        )
    raise AssertionError(f"Non-orthogonal segment: {segment}")


def collinear_overlap_length(first, second):
    (x1, y1), (x2, y2) = first
    (x3, y3), (x4, y4) = second
    if y1 == y2 == y3 == y4:
        return max(
            0,
            min(max(x1, x2), max(x3, x4))
            - max(min(x1, x2), min(x3, x4)),
        )
    if x1 == x2 == x3 == x4:
        return max(
            0,
            min(max(y1, y2), max(y3, y4))
            - max(min(y1, y2), min(y3, y4)),
        )
    return 0


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


class BrandMarkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.marks = []
        self._mark_text = None

    def handle_starttag(self, tag, attrs):
        classes = set(dict(attrs).get("class", "").split())
        if tag == "span" and "brand-mark" in classes:
            self._mark_text = []

    def handle_data(self, data):
        if self._mark_text is not None:
            self._mark_text.append(data)

    def handle_endtag(self, tag):
        if tag == "span" and self._mark_text is not None:
            self.marks.append("".join(self._mark_text).strip())
            self._mark_text = None


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

    def test_published_notes_have_a_single_h1_or_remain_intentionally_empty(self):
        """The reader needs one document title, while empty placeholders must stay empty."""
        for relative_path in EXPECTED_NOTE_PATHS:
            markdown = (DOCS_DIR / relative_path).read_text(encoding="utf-8")
            with self.subTest(relative_path=relative_path):
                if relative_path in EMPTY_NOTE_PATHS:
                    self.assertEqual(markdown, "")
                else:
                    self.assertTrue(markdown.startswith("# "), relative_path)
                    self.assertEqual(len(re.findall(r"(?m)^# ", markdown)), 1)

    def test_published_notes_use_topic_specific_problem_headings(self):
        """A topic heading remains meaningful outside the writing process that created it."""
        for relative_path in EXPECTED_NOTE_PATHS:
            markdown = (DOCS_DIR / relative_path).read_text(encoding="utf-8")
            with self.subTest(relative_path=relative_path):
                self.assertNotIn("这篇笔记解决什么问题", markdown)


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
        for color in ("#FFFFFF", "#8E5F68", "#B98991", "#F4E8EA", "#F7F7F6"):
            with self.subTest(color=color):
                self.assertIn(color, self.styles)

    def test_secondary_surfaces_use_neutral_gray_instead_of_pale_pink(self):
        """Large secondary surfaces should not tint the whole page pink."""
        self.assertIn("--BG-SOFT: #F7F7F6", self.styles)
        self.assertIn("--ACCENT-MIST: #F7F7F6", self.styles)

    def test_every_entry_point_uses_the_single_letter_brand_mark(self):
        """Homepage and note entry points must present the same compact identity."""
        pages = ("index.html", "note.html", "llm/ppo.html", "llm/grpo.html", "llm/concepts.html")
        for page in pages:
            parser = BrandMarkParser()
            parser.feed((DOCS_DIR / page).read_text(encoding="utf-8"))
            with self.subTest(page=page):
                self.assertEqual(parser.marks, ["Z"])

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
                self.assertEqual(parse_qs(urlparse(asset).query).get("v"), ["20260810-3"], asset)


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

    def test_compatibility_pages_do_not_embed_stale_markdown_copies(self):
        """Legacy URLs must render the canonical Markdown instead of a second copy."""
        for page in ("llm/ppo.html", "llm/grpo.html", "llm/concepts.html"):
            with self.subTest(page=page):
                html = (DOCS_DIR / page).read_text(encoding="utf-8")
                self.assertNotIn('id="embedded-markdown"', html)

        note_runtime = (DOCS_DIR / "note.js").read_text(encoding="utf-8")
        self.assertNotIn("embeddedMarkdown", note_runtime)

    def test_viewer_assets_are_versioned_to_prevent_stale_navigation(self):
        """Viewer HTML, scripts, and styles must update as one deployable unit."""
        for page in ("note.html", "llm/ppo.html", "llm/grpo.html", "llm/concepts.html"):
            with self.subTest(page=page):
                parser = self.parse_document(page)
                self.assertTrue(parser.local_assets)
                for asset in parser.local_assets:
                    self.assertEqual(
                        parse_qs(urlparse(asset).query).get("v"),
                        ["20260810-6"],
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
    VLLM_SAMPLING_NOTE_PATH = DOCS_DIR / "infra" / "vllm" / "sampling.md"
    VLLM_SAMPLING_DRAWIO_PATH = (
        DOCS_DIR / "assets" / "infra" / "vllm" / "greedy-sampling-flow.drawio"
    )
    VLLM_SAMPLING_SVG_PATH = (
        DOCS_DIR / "assets" / "infra" / "vllm" / "greedy-sampling-flow.svg"
    )

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
            "## KV Cache 与 Paged Attention 解决的问题",
            "## 三层数据流",
            "## 全局 KV Cache 张量",
            "## BlockManager 生命周期",
            "## 从 `block_table` 到 `slot_mapping`",
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

    def test_greedy_sampling_diagram_has_editable_source_and_accessible_svg(self):
        """The sampling flow must be editable and readable in the note viewer."""
        self.assertTrue(self.VLLM_SAMPLING_DRAWIO_PATH.is_file())
        self.assertTrue(self.VLLM_SAMPLING_SVG_PATH.is_file())

        drawio_root = ET.parse(self.VLLM_SAMPLING_DRAWIO_PATH).getroot()
        self.assertEqual(drawio_root.tag, "mxfile")
        self.assertIsNotNone(drawio_root.find("./diagram/mxGraphModel/root"))

        svg_root = ET.parse(self.VLLM_SAMPLING_SVG_PATH).getroot()
        self.assertEqual(svg_root.attrib.get("viewBox"), "0 0 1200 700")
        self.assertEqual(svg_root.attrib.get("role"), "img")
        self.assertEqual(svg_root.attrib.get("aria-labelledby"), "title desc")
        svg_text = " ".join(svg_root.itertext())
        for label in (
            "temperature",
            "_MAX_TEMP",
            "clamp to 0.01",
            "_SAMPLING_EPS",
            "GREEDY",
            "all_greedy",
            "all_random",
            "argmax",
            "top-k / top-p",
            "torch.where",
        ):
            self.assertIn(label, svg_text)

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
            element.attrib["data-node"]: element for element in node_elements
        }
        edges = {
            element.attrib["data-edge"]: element for element in edge_elements
        }
        return nodes, edges

    def test_greedy_sampling_svg_and_drawio_keep_the_same_topology(self):
        """Published and editable diagrams must describe the same flow graph."""
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
        """Connectors must neither cross unrelated nodes nor overlap each other."""
        nodes, edges = self._greedy_svg_geometry()
        bounds = {
            node_id: svg_node_bounds(node) for node_id, node in nodes.items()
        }
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
                        any(
                            segment_enters_bounds(segment, node_bounds)
                            for segment in edge_segments
                        ),
                        f"{edge_id} enters {node_id}",
                    )

        edge_ids = list(edges)
        for index, first_id in enumerate(edge_ids):
            for second_id in edge_ids[index + 1 :]:
                overlaps = [
                    collinear_overlap_length(first, second)
                    for first in segments[first_id]
                    for second in segments[second_id]
                ]
                self.assertEqual(
                    max(overlaps, default=0),
                    0,
                    f"{first_id} overlaps {second_id}",
                )

    def test_greedy_sampling_arrow_approaches_are_visible(self):
        """Every arrow needs a visible shaft before its marker begins."""
        _, edges = self._greedy_svg_geometry()
        for edge_id, edge in edges.items():
            points = parse_orthogonal_path(edge.attrib["d"])
            (x1, y1), (x2, y2) = points[-2:]
            self.assertGreaterEqual(
                abs(x2 - x1) + abs(y2 - y1),
                36,
                f"{edge_id} has no visible shaft before its arrow",
            )

    def test_sampling_note_embeds_the_greedy_flow(self):
        """The note must connect its explanation to the published flowchart."""
        markdown = self.VLLM_SAMPLING_NOTE_PATH.read_text(encoding="utf-8")
        self.assertTrue(markdown.startswith("# vLLM Sampling\n"))
        self.assertIn("## Sampling 解决的问题", markdown)
        self.assertIn(
            "![vLLM Greedy Sampling 流程](../../assets/infra/vllm/greedy-sampling-flow.svg?v=20260811-1)",
            markdown,
        )
        for identifier in (
            "`temperature`",
            "`_MAX_TEMP`",
            "`_SAMPLING_EPS`",
            "`SamplingType.GREEDY`",
            "`all_greedy`",
            "`all_random`",
            "`torch.where`",
        ):
            self.assertIn(identifier, markdown)

        self.assertIn("原始输入 `temperature = 0`", markdown)
        self.assertIn("原始输入 `1e-6` 不会进入 Greedy", markdown)
        normalization_start = markdown.index("0 < self.temperature < _MAX_TEMP")
        self.assertLess(
            normalization_start,
            markdown.index("if self.temperature < _SAMPLING_EPS", normalization_start),
        )


if __name__ == "__main__":
    unittest.main()
