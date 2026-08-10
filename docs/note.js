const notesToggle = document.querySelector(".notes-toggle");
const tocToggle = document.querySelector(".toc-toggle");
const notesSidebar = document.getElementById("notes-sidebar");
const tocSidebar = document.getElementById("toc-sidebar");
const noteTitle = document.getElementById("note-title");
const noteHeading = document.getElementById("note-heading");
const notePath = document.getElementById("note-path");
const noteContent = document.getElementById("note-content");
const tocNav = document.getElementById("toc-nav");
const notesNav = document.getElementById("notes-nav");
const header = document.querySelector(".site-header");
const embeddedMarkdown = document.getElementById("embedded-markdown");
const sidebarSectionTitle = document.querySelector(".sidebar-section-title");
const notePaths = globalThis.NotePaths;
const codeRendering = globalThis.CodeRendering;

function updateHeaderState() {
  header.classList.toggle("is-scrolled", window.scrollY > 16);
}

function togglePanel(panel, toggle, forceClose = false) {
  if (!panel || !toggle) {
    return;
  }

  const shouldOpen = forceClose ? false : !panel.classList.contains("is-open");
  panel.classList.toggle("is-open", shouldOpen);
  toggle.setAttribute("aria-expanded", String(shouldOpen));
}

function closePanelsOnDesktop() {
  if (window.innerWidth > 1100) {
    togglePanel(notesSidebar, notesToggle, true);
    togglePanel(tocSidebar, tocToggle, true);
  }
}

function slugify(text, usedSlugs) {
  const base = text
    .toLowerCase()
    .trim()
    .replace(/[^\w\u4e00-\u9fff\s-]/g, "")
    .replace(/\s+/g, "-");

  let slug = base || "section";
  let counter = 2;

  while (usedSlugs.has(slug)) {
    slug = `${base || "section"}-${counter}`;
    counter += 1;
  }

  usedSlugs.add(slug);
  return slug;
}

function preprocessMarkdown(markdown) {
  const blockMathPattern = /\\\[\s*([\s\S]*?)\s*\\\]|\$\$\s*([\s\S]*?)\s*\$\$/g;
  const codeBlocks = [];
  const mathBlocks = [];

  function isSimpleInlineMath(content) {
    const normalized = content
      .replace(/\s+/g, " ")
      .trim();

    if (normalized.length === 0 || normalized.length > 24) {
      return false;
    }

    return !/[=+\-/*<>]/.test(normalized) &&
      !/\\frac|\\dfrac|\\cfrac|\\sum|\\prod|\\int|\\min|\\max|\\mathbb|\\left|\\right/.test(normalized);
  }

  function storeCodeBlock(match) {
    const token = `@@CODEBLOCK_${codeBlocks.length}@@`;
    codeBlocks.push(match);
    return token;
  }

  let processedMarkdown = markdown
    .replace(/```[\s\S]*?```/g, storeCodeBlock)
    .replace(/~~~[\s\S]*?~~~/g, storeCodeBlock);

  processedMarkdown = processedMarkdown
    .replace(blockMathPattern, (_, bracketContent, dollarContent) => {
      const content = (bracketContent ?? dollarContent ?? "").trim();
      const normalized = content.replace(/\s+/g, " ").trim();
      if (isSimpleInlineMath(normalized)) {
        return `<span class="inline-math-strong">\\(${escapeHtml(normalized)}\\)</span>`;
      }
      const token = `@@MATHBLOCK_${mathBlocks.length}@@`;
      mathBlocks.push(content);
      return `\n\n${token}\n\n`;
    })
    .replace(/\\\(/g, "$")
    .replace(/\\\)/g, "$");

  codeBlocks.forEach((content, index) => {
    processedMarkdown = processedMarkdown.replace(`@@CODEBLOCK_${index}@@`, content);
  });

  return {
    markdown: processedMarkdown,
    mathBlocks
  };
}

function escapeHtml(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function stripHeadingPrefix(text) {
  return text.replace(/^\s*\d+(?:\.\d+)*[\.\)]?\s+/, "").trim();
}

function getDocumentPrefix() {
  const normalizedPath = window.location.pathname.replace(/\\/g, "/");
  return normalizedPath.endsWith("/note.html") ? "" : "../";
}

function resolveNoteHref(item) {
  return notePaths.createViewerHref(item.path, getDocumentPrefix());
}

function resolveMarkdownFetchPath(path) {
  return `${getDocumentPrefix()}${path}`;
}

function buildNotesNav(activePath) {
  if (!notesNav) {
    return;
  }

  notesNav.innerHTML = "";
  const collection = notePaths.getCollectionForPath(activePath);

  if (!collection) {
    if (sidebarSectionTitle) {
      sidebarSectionTitle.textContent = "Notes";
    }
    notesNav.innerHTML = '<p class="toc-empty">No sibling notes available.</p>';
    return;
  }

  if (sidebarSectionTitle) {
    sidebarSectionTitle.textContent = collection.title;
  }

  collection.items.forEach((item) => {
    const link = document.createElement("a");
    link.className = "note-link";
    link.href = resolveNoteHref(item);
    link.classList.toggle("is-active", item.path === activePath);

    const title = document.createElement("span");
    title.className = "note-link-title";
    title.textContent = item.title;

    const meta = document.createElement("span");
    meta.className = "note-link-meta";
    meta.textContent = item.path.split("/").pop();

    link.append(title, meta);
    notesNav.appendChild(link);
  });
}

function getOutlineHeadings() {
  const preferred = Array.from(noteContent.querySelectorAll("h2, h3, h4"));
  if (preferred.length > 0) {
    return preferred;
  }
  return Array.from(noteContent.querySelectorAll("h1, h2, h3"));
}

function applyHeadingNumbering() {
  const headings = getOutlineHeadings();
  const counters = [0, 0, 0];

  headings.forEach((heading) => {
    const level = Number(heading.tagName.replace("H", ""));
    const normalizedLevel = Math.max(2, Math.min(level, 4));
    const depth = normalizedLevel - 2;

    counters[depth] += 1;
    for (let index = depth + 1; index < counters.length; index += 1) {
      counters[index] = 0;
    }

    const numberLabel = counters.slice(0, depth + 1).filter(Boolean).join(".");
    const titleText = heading.dataset.headingText || heading.textContent.trim();

    heading.dataset.headingText = titleText;
    heading.dataset.headingNumber = numberLabel;
    heading.innerHTML = "";

    const number = document.createElement("span");
    number.className = "heading-number";
    number.textContent = `${numberLabel}`;

    const text = document.createElement("span");
    text.className = "heading-text";
    text.textContent = titleText;

    heading.append(number, text);
  });
}

function buildToc() {
  const headings = getOutlineHeadings();

  if (headings.length === 0) {
    tocNav.innerHTML = '<p class="toc-empty">No headings found in this note.</p>';
    return;
  }

  tocNav.innerHTML = "";

  headings.forEach((heading) => {
    const link = document.createElement("a");
    const level = Number(heading.tagName.replace("H", ""));
    const visibleLevel = Math.max(2, Math.min(level, 4));
    const label = heading.dataset.headingText || heading.textContent.trim();

    link.className = `toc-link level-${visibleLevel}`;
    link.href = `#${heading.id}`;

    if (heading.dataset.headingNumber) {
      const number = document.createElement("span");
      number.className = "toc-number";
      number.textContent = heading.dataset.headingNumber;
      link.appendChild(number);
    }

    const text = document.createElement("span");
    text.className = "toc-text";
    text.textContent = label;
    link.appendChild(text);

    link.addEventListener("click", () => {
      if (window.innerWidth <= 1100) {
        togglePanel(tocSidebar, tocToggle, true);
      }
    });

    tocNav.appendChild(link);
  });

  const tocLinks = Array.from(tocNav.querySelectorAll(".toc-link"));
  const observer = new IntersectionObserver(
    (entries) => {
      const activeEntry = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

      if (!activeEntry) {
        return;
      }

      tocLinks.forEach((link) => {
        link.classList.toggle("is-active", link.getAttribute("href") === `#${activeEntry.target.id}`);
      });
    },
    {
      rootMargin: "-20% 0px -65% 0px",
      threshold: [0.1, 0.35, 0.6]
    }
  );

  headings.forEach((heading) => observer.observe(heading));
}

function renderMath() {
  if (typeof renderMathInElement !== "function") {
    return;
  }

  renderMathInElement(noteContent, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "\\[", right: "\\]", display: true },
      { left: "\\(", right: "\\)", display: false },
      { left: "$", right: "$", display: false }
    ],
    ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
    throwOnError: false
  });

  normalizeDisplayMath();
}

function normalizeDisplayMath() {
  noteContent.querySelectorAll(".katex-display").forEach((displayMath) => {
    const parent = displayMath.parentElement;
    if (!parent) {
      return;
    }

    parent.classList.add("math-block");
    parent.classList.add("math-block-display");
  });
}

function highlightCodeBlocks() {
  noteContent.querySelectorAll("pre code").forEach((block) => {
    if (codeRendering?.decorateCodeBlock) {
      codeRendering.decorateCodeBlock(
        block,
        typeof hljs === "undefined" ? undefined : hljs
      );
    } else if (typeof hljs !== "undefined") {
      block.classList.add("hljs");
      hljs.highlightElement(block);
    }
    enhanceCodeHighlight(block);
  });
}

function enhanceCodeHighlight(block) {
  const languageClass = Array.from(block.classList)
    .find((className) => className.startsWith("language-"));
  const language = languageClass?.replace("language-", "");

  if (!["python", "py"].includes(language)) {
    return;
  }

  const walker = document.createTreeWalker(block, NodeFilter.SHOW_TEXT);
  const textNodes = [];

  while (walker.nextNode()) {
    const node = walker.currentNode;
    const parentClass = node.parentElement?.className || "";
    if (!/hljs-(keyword|string|comment|number|literal|built_in|title|function|class|meta|attr|params)/.test(parentClass)) {
      textNodes.push(node);
    }
  }

  textNodes.forEach((node) => {
    const fragment = document.createDocumentFragment();
    const parts = node.textContent.split(/(\b(?:self|cls)\b|\b[a-zA-Z_]\w*(?=\s*=)|(?<=\.)[a-zA-Z_]\w*)/g);

    parts.forEach((part) => {
      if (!part) {
        return;
      }

      if (/^(self|cls)$/.test(part)) {
        const span = document.createElement("span");
        span.className = "hljs-variable language-self";
        span.textContent = part;
        fragment.appendChild(span);
      } else if (/^[a-zA-Z_]\w*$/.test(part)) {
        const span = document.createElement("span");
        span.className = "hljs-variable";
        span.textContent = part;
        fragment.appendChild(span);
      } else {
        fragment.appendChild(document.createTextNode(part));
      }
    });

    node.replaceWith(fragment);
  });
}

function renderMarkdown(markdown, path) {
  const normalizedContent = markdown.trim().length > 0
    ? preprocessMarkdown(markdown)
    : {
        markdown: "# Empty note\n\nThis markdown file exists, but it does not contain content yet.",
        mathBlocks: []
      };

  if (typeof marked?.parse !== "function") {
    noteContent.innerHTML = "";
    const fallback = document.createElement("pre");
    fallback.textContent = normalizedContent.markdown;
    noteContent.appendChild(fallback);
    noteTitle.textContent = path.split("/").pop();
    noteHeading.textContent = "Markdown preview unavailable";
    document.title = `${noteTitle.textContent} | Note Viewer`;
    return;
  }

  let renderedHtml = marked.parse(normalizedContent.markdown, {
    gfm: true,
    breaks: false
  });

  normalizedContent.mathBlocks.forEach((content, index) => {
    const token = `@@MATHBLOCK_${index}@@`;
    const blockHtml = `<div class="math-block">\\[\n${escapeHtml(content)}\n\\]</div>`;
    renderedHtml = renderedHtml
      .replace(`<p>${token}</p>`, blockHtml)
      .replace(token, blockHtml);
  });

  noteContent.innerHTML = renderedHtml;

  const usedSlugs = new Set();
  const headings = Array.from(noteContent.querySelectorAll("h1, h2, h3, h4, h5, h6"));
  headings.forEach((heading) => {
    heading.id = slugify(heading.textContent, usedSlugs);
    heading.dataset.headingText = stripHeadingPrefix(heading.textContent.trim());
  });

  const firstHeading = headings[0]?.dataset.headingText || path.split("/").pop().replace(".md", "");
  noteTitle.textContent = firstHeading;
  noteHeading.textContent = firstHeading;
  document.title = `${firstHeading} | Note Viewer`;

  applyHeadingNumbering();
  highlightCodeBlocks();
  renderMath();
  buildToc();
}

function showNoteError(title, detail) {
  noteTitle.textContent = title;
  noteHeading.textContent = title;
  noteContent.innerHTML = "";

  const message = document.createElement("p");
  message.className = "note-error";
  message.textContent = detail;
  noteContent.appendChild(message);

  tocNav.innerHTML = '<p class="toc-empty">No table of contents available.</p>';
  document.title = `${title} | Note Viewer`;
}

async function loadNote() {
  const params = new URLSearchParams(window.location.search);
  const requestedPath = body.dataset.notePath || params.get("path");

  if (!notePaths) {
    showNoteError("Unable to load note", "The note path module is unavailable.");
    return;
  }

  const path = notePaths.sanitizeNotePath(requestedPath);
  if (!path) {
    notePath.textContent = requestedPath || "";
    buildNotesNav(null);
    showNoteError("Invalid note path", "The requested Markdown path is not allowed.");
    return;
  }

  notePath.textContent = path;
  buildNotesNav(path);

  try {
    const response = await fetch(resolveMarkdownFetchPath(path), { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Failed to load ${path}`);
    }

    const rawMarkdown = await response.text();
    renderMarkdown(rawMarkdown, path);
  } catch (error) {
    if (embeddedMarkdown) {
      renderMarkdown(embeddedMarkdown.textContent, path);
      return;
    }

    const detail = window.location.protocol === "file:"
      ? `${error.message}. Local file previews cannot fetch Markdown. Open a note page with embedded content or run a local web server.`
      : error.message;
    showNoteError("Unable to load note", detail);
  }
}

notesToggle?.addEventListener("click", () => {
  togglePanel(notesSidebar, notesToggle);
  togglePanel(tocSidebar, tocToggle, true);
});

tocToggle?.addEventListener("click", () => {
  togglePanel(tocSidebar, tocToggle);
  togglePanel(notesSidebar, notesToggle, true);
});

document.addEventListener("click", (event) => {
  if (window.innerWidth > 1100) {
    return;
  }

  const clickedNotesToggle = notesToggle?.contains(event.target);
  const clickedTocToggle = tocToggle?.contains(event.target);
  const insideNotes = notesSidebar?.contains(event.target);
  const insideToc = tocSidebar?.contains(event.target);

  if (!clickedNotesToggle && !insideNotes) {
    togglePanel(notesSidebar, notesToggle, true);
  }

  if (!clickedTocToggle && !insideToc) {
    togglePanel(tocSidebar, tocToggle, true);
  }
});

window.addEventListener("scroll", updateHeaderState, { passive: true });
window.addEventListener("resize", closePanelsOnDesktop);

updateHeaderState();
loadNote();
