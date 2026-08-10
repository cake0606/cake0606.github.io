const test = require("node:test");
const assert = require("node:assert/strict");

const NotePaths = require("../docs/note-paths.js");


test("normalizes approved relative Markdown paths", () => {
  assert.equal(
    NotePaths.sanitizeNotePath("infra/vllm/scheduler.md"),
    "infra/vllm/scheduler.md"
  );
  assert.equal(
    NotePaths.sanitizeNotePath("infra/cuda/基础.md"),
    "infra/cuda/基础.md"
  );
  assert.equal(NotePaths.sanitizeNotePath(), "llm/rl/ppo.md");
});

test("rejects unsafe or unsupported note paths", () => {
  const invalidPaths = [
    "../secret.md",
    "llm/../secret.md",
    "/llm/rl/ppo.md",
    "https://example.com/a.md",
    "infra\\vllm\\scheduler.md",
    "infra/vllm/scheduler.txt",
    "infra//vllm/scheduler.md"
  ];

  invalidPaths.forEach((path) => {
    assert.equal(NotePaths.sanitizeNotePath(path), null, path);
  });
});

test("returns the sibling collection for an organized note", () => {
  assert.deepEqual(
    NotePaths.getCollectionForPath("llm/rl/ppo.md").items.map((item) => item.path),
    ["llm/rl/ppo.md", "llm/rl/grpo.md"]
  );
  assert.deepEqual(
    NotePaths.getCollectionForPath("infra/vllm/scheduler.md").items.map((item) => item.path),
    ["infra/vllm/sampling.md", "infra/vllm/scheduler.md"]
  );
  assert.equal(NotePaths.getCollectionForPath("unlisted/note.md"), null);
});

test("builds encoded viewer links at root and compatibility-page depth", () => {
  assert.equal(
    NotePaths.createViewerHref("infra/vllm/scheduler.md", ""),
    "note.html?path=infra%2Fvllm%2Fscheduler.md"
  );
  assert.equal(
    NotePaths.createViewerHref("llm/rl/ppo.md", "../"),
    "../note.html?path=llm%2Frl%2Fppo.md"
  );
  assert.equal(NotePaths.createViewerHref("../secret.md", ""), null);
});
