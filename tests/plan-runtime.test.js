const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const source = fs.readFileSync(
  path.join(__dirname, "..", "docs", "script.js"),
  "utf8"
);

function createElement(tagName = "div") {
  const classes = new Set();
  const element = {
    tagName: tagName.toUpperCase(),
    children: [],
    attributes: {},
    hidden: false,
    textContent: "",
    classList: {
      add(...names) { names.forEach((name) => classes.add(name)); },
      contains(name) { return classes.has(name); },
      toggle(name, force) {
        const enabled = force === undefined ? !classes.has(name) : Boolean(force);
        if (enabled) classes.add(name);
        else classes.delete(name);
        return enabled;
      }
    },
    append(...nodes) { this.children.push(...nodes); },
    replaceChildren(...nodes) { this.children = [...nodes]; },
    setAttribute(name, value) { this.attributes[name] = String(value); },
    getAttribute(name) { return this.attributes[name]; },
    addEventListener() {}
  };

  Object.defineProperty(element, "className", {
    get() { return [...classes].join(" "); },
    set(value) {
      classes.clear();
      String(value).split(/\s+/).filter(Boolean).forEach((name) => classes.add(name));
    }
  });

  return element;
}

function createHarness(fetchImpl) {
  const header = createElement("header");
  const planGroups = createElement("div");
  const planState = createElement("p");
  const errors = [];
  const fetchCalls = [];

  const context = {
    console: { error(...args) { errors.push(args); } },
    document: {
      createElement,
      getElementById(id) {
        return id === "plan-groups" ? planGroups : id === "plan-state" ? planState : null;
      },
      querySelector(selector) { return selector === ".site-header" ? header : null; },
      querySelectorAll() { return []; }
    },
    fetch(url, options) {
      fetchCalls.push({ url, options });
      return fetchImpl(url, options);
    },
    window: {
      addEventListener() {},
      location: { hash: "" },
      scrollY: 0
    }
  };
  context.globalThis = context;
  vm.runInNewContext(source, context);

  return {
    errors,
    fetchCalls,
    header,
    planGroups,
    planState,
    async settle() {
      await new Promise((resolve) => setImmediate(resolve));
      await new Promise((resolve) => setImmediate(resolve));
    }
  };
}

test("loads grouped plans without caching and renders semantic status", async () => {
  const data = [
    {
      title: "infra",
      items: [
        { text: "prefill与decode", completed: false },
        { text: "cuda graph", completed: true }
      ]
    },
    { title: "llm", items: [{ text: "mha、mqa、gqa", completed: false }] }
  ];
  const harness = createHarness(async () => ({ ok: true, json: async () => data }));
  await harness.settle();

  assert.equal(harness.fetchCalls.length, 1);
  assert.equal(harness.fetchCalls[0].url, "plan.json");
  assert.equal(harness.fetchCalls[0].options.cache, "no-store");
  assert.equal(harness.planGroups.getAttribute("aria-busy"), "false");
  assert.equal(harness.planState.hidden, true);
  assert.equal(harness.planGroups.children.length, 2);

  const [infra, llm] = harness.planGroups.children;
  assert.equal(infra.tagName, "SECTION");
  assert.equal(infra.children[0].tagName, "H3");
  assert.equal(infra.children[0].textContent, "infra");
  assert.equal(llm.children[0].textContent, "llm");

  const items = infra.children[1].children;
  assert.equal(items[0].children[0].getAttribute("aria-label"), "未完成");
  assert.equal(items[0].children[1].textContent, "prefill与decode");
  assert.equal(items[1].classList.contains("is-complete"), true);
  assert.equal(items[1].children[0].getAttribute("aria-label"), "已完成");
});

test("shows the empty state for a valid empty plan", async () => {
  const harness = createHarness(async () => ({ ok: true, json: async () => [] }));
  await harness.settle();
  assert.equal(harness.planGroups.children.length, 0);
  assert.equal(harness.planGroups.getAttribute("aria-busy"), "false");
  assert.equal(harness.planState.hidden, false);
  assert.equal(harness.planState.textContent, "No plans published yet.");
});

test("shows one non-fatal error state for request, parse, HTTP, and schema failures", async (t) => {
  const cases = [
    ["request", async () => { throw new Error("offline"); }],
    ["HTTP", async () => ({ ok: false, status: 404, json: async () => [] })],
    ["parse", async () => ({ ok: true, json: async () => { throw new SyntaxError("bad json"); } })],
    ["schema", async () => ({ ok: true, json: async () => [{ title: "", items: [] }] })]
  ];

  for (const [name, fetchImpl] of cases) {
    await t.test(name, async () => {
      const harness = createHarness(fetchImpl);
      await harness.settle();
      assert.equal(harness.planGroups.children.length, 0);
      assert.equal(harness.planGroups.getAttribute("aria-busy"), "false");
      assert.equal(harness.planState.textContent, "Plans are temporarily unavailable.");
      assert.equal(harness.planState.hidden, false);
      assert.equal(harness.errors.length, 1);
    });
  }
});
