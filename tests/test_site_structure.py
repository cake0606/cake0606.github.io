from pathlib import Path
import unittest


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


if __name__ == "__main__":
    unittest.main()
