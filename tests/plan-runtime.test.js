const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const source = fs.readFileSync(
  path.join(__dirname, "..", "docs", "script.js"),
  "utf8"
);
const styles = fs.readFileSync(
  path.join(__dirname, "..", "docs", "style.css"),
  "utf8"
);

function getCssDeclarations(selector) {
  const escapedSelector = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = styles.match(new RegExp(`${escapedSelector}\\s*\\{([^}]*)\\}`));
  assert.ok(match, `Missing CSS rule for ${selector}`);

  return Object.fromEntries(
    match[1]
      .split(";")
      .map((declaration) => declaration.trim())
      .filter(Boolean)
      .map((declaration) => declaration.split(":").map((part) => part.trim()))
  );
}

function createElement(tagName = "div") {
  const classes = new Set();
  const element = {
    tagName: tagName.toUpperCase(),
    children: [],
    attributes: {},
    hidden: false,
    listeners: [],
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
    addEventListener(type, listener, options) {
      this.listeners.push({ type, listener, options });
    }
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
  const navLink = createElement("a");
  const projects = createElement("section");
  const planGroups = createElement("div");
  const planState = createElement("p");
  const errors = [];
  const fetchCalls = [];
  const observedSections = [];
  const windowListeners = [];

  navLink.className = "nav-link";
  navLink.setAttribute("href", "#projects");
  projects.id = "projects";

  const context = {
    console: { error(...args) { errors.push(args); } },
    document: {
      createElement,
      getElementById(id) {
        return id === "plan-groups" ? planGroups : id === "plan-state" ? planState : null;
      },
      querySelector(selector) {
        return selector === ".site-header" ? header : selector === "#projects" ? projects : null;
      },
      querySelectorAll(selector) { return selector === ".nav-link" ? [navLink] : []; }
    },
    fetch(url, options) {
      fetchCalls.push({ url, options });
      return fetchImpl(url, options);
    },
    IntersectionObserver: class {
      observe(section) { observedSections.push(section); }
    },
    window: {
      addEventListener(type, listener, options) {
        windowListeners.push({ type, listener, options });
      },
      location: { hash: "" },
      scrollY: 9
    }
  };
  context.globalThis = context;
  vm.runInNewContext(source, context);

  return {
    errors,
    fetchCalls,
    header,
    navLink,
    observedSections,
    planGroups,
    planState,
    windowListeners,
    async settle() {
      await new Promise((resolve) => setImmediate(resolve));
      await new Promise((resolve) => setImmediate(resolve));
    }
  };
}

function assertNavigationInitialized(harness) {
  assert.equal(harness.header.classList.contains("is-scrolled"), true);
  assert.equal(harness.navLink.classList.contains("is-active"), true);
  assert.equal(harness.navLink.listeners.filter(({ type }) => type === "click").length, 1);
  assert.equal(harness.observedSections.length, 1);
  assert.equal(harness.windowListeners.filter(({ type }) => type === "scroll").length, 1);
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

  const infraList = infra.children[1];
  const llmList = llm.children[1];
  assert.equal(infraList.tagName, "UL");
  assert.equal(infraList.getAttribute("aria-label"), "infra plan");
  assert.equal(llmList.tagName, "UL");
  assert.equal(llmList.getAttribute("aria-label"), "llm plan");

  const items = infraList.children;
  assert.equal(items[0].tagName, "LI");
  assert.equal(items[1].tagName, "LI");
  assert.equal(llmList.children[0].tagName, "LI");
  assert.equal(items[0].children[0].getAttribute("aria-label"), "未完成");
  assert.equal(items[0].children[1].textContent, "prefill与decode");
  assert.equal(items[1].classList.contains("is-complete"), true);
  assert.equal(items[1].children[0].getAttribute("aria-label"), "已完成");
});

test("keeps long unbroken plan text shrinkable and wrappable", async () => {
  const longToken = "long-plan-token-".repeat(40);
  const data = [{ title: "infra", items: [{ text: longToken, completed: false }] }];
  const harness = createHarness(async () => ({ ok: true, json: async () => data }));
  await harness.settle();

  const text = harness.planGroups.children[0].children[1].children[0].children[1];
  assert.equal(text.classList.contains("plan-text"), true);
  assert.equal(text.textContent, longToken);

  const declarations = getCssDeclarations(".plan-text");
  assert.equal(declarations["min-width"], "0");
  assert.equal(declarations["overflow-wrap"], "anywhere");
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
      assertNavigationInitialized(harness);
    });
  }
});
