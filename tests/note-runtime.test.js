const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const NotePaths = require("../docs/note-paths.js");

function createElement() {
  return {
    classList: {
      add() {},
      toggle() {},
      contains() { return false; }
    },
    dataset: {},
    append() {},
    appendChild() {},
    addEventListener() {},
    contains() { return false; },
    querySelectorAll() { return []; },
    setAttribute() {},
    innerHTML: "",
    textContent: ""
  };
}

test("starts loading the Markdown selected by the root viewer query string", () => {
  const source = fs.readFileSync(
    path.join(__dirname, "..", "docs", "note.js"),
    "utf8"
  );
  const fetchCalls = [];
  const elements = new Map();
  const getElement = (key) => {
    if (!elements.has(key)) {
      elements.set(key, createElement());
    }
    return elements.get(key);
  };

  const document = {
    body: { dataset: {} },
    title: "",
    createElement,
    createDocumentFragment: createElement,
    createTextNode: (text) => ({ textContent: text }),
    getElementById: getElement,
    querySelector: getElement,
    addEventListener() {}
  };
  const window = {
    innerWidth: 1440,
    scrollY: 0,
    location: {
      pathname: "/note.html",
      search: "?path=infra/nano-vllm/kvcache_and_paged_attention.md",
      protocol: "http:"
    },
    addEventListener() {}
  };

  assert.doesNotThrow(() => {
    vm.runInNewContext(source, {
      CodeRendering: {},
      NotePaths,
      URLSearchParams,
      document,
      fetch: (url, options) => {
        fetchCalls.push({ url, options });
        return new Promise(() => {});
      },
      globalThis: { CodeRendering: {}, NotePaths },
      window
    });
  });

  assert.equal(fetchCalls.length, 1);
  assert.equal(
    fetchCalls[0].url,
    "infra/nano-vllm/kvcache_and_paged_attention.md"
  );
  assert.equal(fetchCalls[0].options.cache, "no-store");
});
