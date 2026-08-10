(function exposeNotePaths(root, factory) {
  const api = factory();
  root.NotePaths = api;

  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : window, function createNotePaths() {
  const DEFAULT_NOTE_PATH = "llm/rl/ppo.md";

  const NOTE_COLLECTIONS = {
    "infra/cuda": {
      title: "CUDA",
      items: [
        { title: "CUDA Memory", path: "infra/cuda/cuda内存.md" },
        { title: "Elementwise", path: "infra/cuda/elementwise.md" },
        { title: "Dot, Softmax & Norm", path: "infra/cuda/点乘_softmax_norm.md" },
        { title: "CUDA Basics", path: "infra/cuda/基础.md" }
      ]
    },
    "infra/nano-vllm": {
      title: "nano-vLLM",
      items: [
        { title: "KV Cache & Paged Attention", path: "infra/nano-vllm/kvcache_and_paged_attention.md" },
        { title: "LLM Engine", path: "infra/nano-vllm/llm_engine.md" },
        { title: "Model Runner", path: "infra/nano-vllm/ModelRunner.md" },
        { title: "Run nano-vLLM", path: "infra/nano-vllm/run_nano-vllm.md" },
        { title: "Scheduler", path: "infra/nano-vllm/Scheduler.md" },
        { title: "Structure", path: "infra/nano-vllm/struct.md" }
      ]
    },
    "infra/vllm": {
      title: "vLLM",
      items: [
        { title: "Sampling", path: "infra/vllm/sampling.md" },
        { title: "Scheduler", path: "infra/vllm/scheduler.md" }
      ]
    },
    "llm/rl": {
      title: "Reinforcement Learning",
      items: [
        { title: "PPO", path: "llm/rl/ppo.md" },
        { title: "GRPO", path: "llm/rl/grpo.md" }
      ]
    },
    "llm/concepts": {
      title: "LLM Concepts",
      items: [
        { title: "Concepts", path: "llm/concepts/concepts.md" },
        { title: "GAE", path: "llm/concepts/gae.md" },
        { title: "Index", path: "llm/concepts/index.md" },
        { title: "KV Cache", path: "llm/concepts/kvcache.md" }
      ]
    }
  };

  function sanitizeNotePath(rawPath) {
    if (rawPath === undefined || rawPath === null || rawPath === "") {
      return DEFAULT_NOTE_PATH;
    }

    if (typeof rawPath !== "string") {
      return null;
    }

    const path = rawPath.trim();
    const segments = path.split("/");
    const hasUnsafeSegment = segments.some(
      (segment) => segment === "" || segment === "." || segment === ".."
    );

    if (
      path === "" ||
      path.includes("\\") ||
      path.startsWith("/") ||
      /^[a-z][a-z\d+.-]*:/i.test(path) ||
      hasUnsafeSegment ||
      !path.toLowerCase().endsWith(".md")
    ) {
      return null;
    }

    return path;
  }

  function getCollectionForPath(rawPath) {
    const path = sanitizeNotePath(rawPath);
    if (!path) {
      return null;
    }

    const separator = path.lastIndexOf("/");
    if (separator < 0) {
      return null;
    }

    return NOTE_COLLECTIONS[path.slice(0, separator)] || null;
  }

  function createViewerHref(rawPath, prefix = "") {
    const path = sanitizeNotePath(rawPath);
    if (!path) {
      return null;
    }

    return `${prefix}note.html?path=${encodeURIComponent(path)}`;
  }

  function resolveNoteAssetHref(rawNotePath, rawHref, prefix = "") {
    const notePath = sanitizeNotePath(rawNotePath);
    if (!notePath || typeof rawHref !== "string" || typeof prefix !== "string") {
      return null;
    }

    const href = rawHref.trim();
    if (!href) {
      return null;
    }

    if (
      href.startsWith("/") ||
      href.startsWith("#") ||
      href.startsWith("?") ||
      /^[a-z][a-z\d+.-]*:/i.test(href)
    ) {
      return href;
    }

    const suffixIndex = href.search(/[?#]/);
    const relativePath = suffixIndex === -1 ? href : href.slice(0, suffixIndex);
    const suffix = suffixIndex === -1 ? "" : href.slice(suffixIndex);
    if (!relativePath || relativePath.includes("\\")) {
      return null;
    }

    const resolvedSegments = notePath.split("/").slice(0, -1);
    for (const segment of relativePath.split("/")) {
      if (!segment || segment === ".") {
        continue;
      }
      if (segment === "..") {
        if (resolvedSegments.length === 0) {
          return null;
        }
        resolvedSegments.pop();
        continue;
      }
      resolvedSegments.push(segment);
    }

    if (resolvedSegments.length === 0) {
      return null;
    }

    return `${prefix}${resolvedSegments.join("/")}${suffix}`;
  }

  return {
    DEFAULT_NOTE_PATH,
    NOTE_COLLECTIONS,
    sanitizeNotePath,
    getCollectionForPath,
    createViewerHref,
    resolveNoteAssetHref
  };
});
