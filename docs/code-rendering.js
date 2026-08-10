(function attachCodeRendering(globalObject) {
  const LANGUAGE_ALIASES = Object.freeze({
    bash: { highlightLanguage: "bash", label: "Bash" },
    c: { highlightLanguage: "c", label: "C" },
    "c++": { highlightLanguage: "cpp", label: "C++" },
    cpp: { highlightLanguage: "cpp", label: "C++" },
    cu: { highlightLanguage: "cpp", label: "CUDA" },
    cuda: { highlightLanguage: "cpp", label: "CUDA" },
    cxx: { highlightLanguage: "cpp", label: "C++" },
    css: { highlightLanguage: "css", label: "CSS" },
    go: { highlightLanguage: "go", label: "Go" },
    html: { highlightLanguage: "xml", label: "HTML" },
    java: { highlightLanguage: "java", label: "Java" },
    javascript: { highlightLanguage: "javascript", label: "JavaScript" },
    js: { highlightLanguage: "javascript", label: "JavaScript" },
    json: { highlightLanguage: "json", label: "JSON" },
    jsonc: { highlightLanguage: "json", label: "JSON" },
    markdown: { highlightLanguage: "markdown", label: "Markdown" },
    md: { highlightLanguage: "markdown", label: "Markdown" },
    plaintext: { highlightLanguage: null, label: "Text" },
    powershell: { highlightLanguage: "powershell", label: "PowerShell" },
    ps1: { highlightLanguage: "powershell", label: "PowerShell" },
    py: { highlightLanguage: "python", label: "Python" },
    python: { highlightLanguage: "python", label: "Python" },
    rust: { highlightLanguage: "rust", label: "Rust" },
    sh: { highlightLanguage: "bash", label: "Bash" },
    shell: { highlightLanguage: "bash", label: "Bash" },
    sql: { highlightLanguage: "sql", label: "SQL" },
    text: { highlightLanguage: null, label: "Text" },
    ts: { highlightLanguage: "typescript", label: "TypeScript" },
    txt: { highlightLanguage: null, label: "Text" },
    typescript: { highlightLanguage: "typescript", label: "TypeScript" },
    xml: { highlightLanguage: "xml", label: "XML" },
    yaml: { highlightLanguage: "yaml", label: "YAML" },
    yml: { highlightLanguage: "yaml", label: "YAML" },
    zsh: { highlightLanguage: "bash", label: "Zsh" }
  });

  function titleCaseIdentifier(identifier) {
    return identifier
      .split(/[-_]+/)
      .filter(Boolean)
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(" ") || "Text";
  }

  function resolveLanguage(classNames = []) {
    const languageClass = Array.from(classNames)
      .find((className) => className.startsWith("language-"));
    const sourceLanguage = languageClass
      ? languageClass.slice("language-".length).toLowerCase()
      : "text";
    const knownLanguage = LANGUAGE_ALIASES[sourceLanguage];

    return {
      sourceLanguage,
      highlightLanguage: knownLanguage?.highlightLanguage ?? null,
      label: knownLanguage?.label ?? titleCaseIdentifier(sourceLanguage)
    };
  }

  function decorateCodeBlock(block, highlighter) {
    const metadata = resolveLanguage(block?.classList);
    const pre = block?.parentElement;

    if (!block || !pre) {
      return metadata;
    }

    pre.dataset.language = metadata.label;
    pre.classList.add("has-language-label");

    Array.from(block.classList)
      .filter((className) => className.startsWith("language-"))
      .forEach((className) => block.classList.remove(className));

    block.classList.add("hljs");

    const canHighlight = Boolean(
      metadata.highlightLanguage &&
      highlighter &&
      typeof highlighter.highlightElement === "function" &&
      (typeof highlighter.getLanguage !== "function" || highlighter.getLanguage(metadata.highlightLanguage))
    );

    if (!canHighlight) {
      block.classList.add("nohighlight");
      return metadata;
    }

    block.classList.add(`language-${metadata.highlightLanguage}`);
    highlighter.highlightElement(block);
    return metadata;
  }

  const api = Object.freeze({
    decorateCodeBlock,
    resolveLanguage
  });

  globalObject.CodeRendering = api;

  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : window);
