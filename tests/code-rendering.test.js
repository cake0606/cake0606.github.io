const assert = require("node:assert/strict");
const test = require("node:test");

const {
  decorateCodeBlock,
  resolveLanguage
} = require("../docs/code-rendering.js");

function createClassList(initialClasses = []) {
  const values = new Set(initialClasses);
  return {
    add(...classes) {
      classes.forEach((className) => values.add(className));
    },
    remove(...classes) {
      classes.forEach((className) => values.delete(className));
    },
    contains(className) {
      return values.has(className);
    },
    [Symbol.iterator]() {
      return values[Symbol.iterator]();
    }
  };
}

function createBlock(classes) {
  const pre = {
    classList: createClassList(),
    dataset: {}
  };
  return {
    classList: createClassList(classes),
    parentElement: pre,
    pre
  };
}

test("normalizes Python, CUDA, Bash, and text identifiers", () => {
  assert.deepEqual(resolveLanguage(["language-py"]), {
    sourceLanguage: "py",
    highlightLanguage: "python",
    label: "Python"
  });
  assert.deepEqual(resolveLanguage(["language-cuda"]), {
    sourceLanguage: "cuda",
    highlightLanguage: "cpp",
    label: "CUDA"
  });
  assert.deepEqual(resolveLanguage(["language-shell"]), {
    sourceLanguage: "shell",
    highlightLanguage: "bash",
    label: "Bash"
  });
  assert.deepEqual(resolveLanguage(["language-text"]), {
    sourceLanguage: "text",
    highlightLanguage: null,
    label: "Text"
  });
});

test("falls back to a readable label without guessing an unknown highlighter", () => {
  assert.deepEqual(resolveLanguage(["language-tensor-shape"]), {
    sourceLanguage: "tensor-shape",
    highlightLanguage: null,
    label: "Tensor Shape"
  });
  assert.deepEqual(resolveLanguage([]), {
    sourceLanguage: "text",
    highlightLanguage: null,
    label: "Text"
  });
});

test("decorates and highlights a supported alias", () => {
  const block = createBlock(["language-cuda"]);
  const highlighted = [];
  const highlighter = {
    getLanguage(language) {
      return language === "cpp";
    },
    highlightElement(element) {
      highlighted.push(element);
    }
  };

  const result = decorateCodeBlock(block, highlighter);

  assert.equal(result.label, "CUDA");
  assert.equal(block.pre.dataset.language, "CUDA");
  assert.equal(block.pre.classList.contains("has-language-label"), true);
  assert.equal(block.classList.contains("language-cuda"), false);
  assert.equal(block.classList.contains("language-cpp"), true);
  assert.equal(block.classList.contains("hljs"), true);
  assert.deepEqual(highlighted, [block]);
});

test("keeps plain and unavailable languages readable without throwing", () => {
  const textBlock = createBlock(["language-text"]);
  const unknownBlock = createBlock(["language-raku"]);
  const highlighter = {
    getLanguage() {
      return false;
    },
    highlightElement() {
      throw new Error("must not highlight unsupported code");
    }
  };

  decorateCodeBlock(textBlock, highlighter);
  decorateCodeBlock(unknownBlock, highlighter);

  assert.equal(textBlock.classList.contains("nohighlight"), true);
  assert.equal(unknownBlock.classList.contains("nohighlight"), true);
  assert.equal(textBlock.pre.dataset.language, "Text");
  assert.equal(unknownBlock.pre.dataset.language, "Raku");
});
