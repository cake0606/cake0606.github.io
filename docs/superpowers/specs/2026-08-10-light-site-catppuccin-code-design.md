# Light-only site and Catppuccin code rendering design

## Goal

Keep the website permanently in its approved light presentation while making inline code and fenced code visually distinct. Inline code should use the site's Morandi pink palette. Fenced code should use Catppuccin Mocha syntax highlighting and display an explicit language label.

## Scope

The change applies to the homepage, the generic Markdown viewer, and the three retained compatibility note pages. Markdown prose and technical meaning remain unchanged; only code-fence language identifiers may be normalized or added.

## Theme architecture

The site will have one global theme: the existing white and Morandi-pink light palette. The dark variable set, `data-theme` branches, theme buttons, saved theme preference, and system color-scheme detection will be removed. Header and responsive styles will reference the light variables directly.

Code blocks are intentionally independent of the site theme. They will always use Catppuccin Mocha so that syntax colors have consistent contrast against a dark editor-like surface.

## Code rendering

The current Highlight.js core script will be replaced by the documented full browser bundle for Highlight.js 11.11.1. The Catppuccin Mocha Highlight.js stylesheet will be stored locally in `docs/vendor/` so its appearance does not depend on a theme CDN.

After Marked renders Markdown, the viewer will inspect every `pre > code` element:

1. Read its Markdown language class.
2. Normalize aliases before highlighting: `py` to `python`, `cu` or `cuda` to the C++ highlighter, and shell aliases to `bash`.
3. Preserve a reader-friendly label such as `Python`, `CUDA`, `Bash`, `JSON`, or `Text`.
4. Run Highlight.js.
5. Add a positioned language label to the containing `pre` element.

If Highlight.js is unavailable, the original code remains readable and the language label is still displayed. Unknown identifiers fall back to escaped plain text rather than breaking the note.

## Markdown normalization

All fenced blocks under `docs/infra/` and `docs/llm/` will be audited. Existing short aliases will be normalized to descriptive identifiers where appropriate. The two currently unlabeled blocks in `kvcache_and_paged_attention.md` and `ModelRunner.md` will be marked as `text`. No prose or executable examples will be rewritten.

Python triple-quoted strings inside fenced Python examples remain content, not Markdown delimiters. Inline snippets continue to use Markdown backticks.

## Visual design

Inline code will use the site's light Morandi treatment: soft pink background, muted pink border, and deep rose text. It will no longer use the blue-gray treatment.

Fenced blocks will use Catppuccin Mocha base colors, including the official dark base, text, mauve, blue, green, peach, pink, teal, and overlay colors. The language label will sit at the upper-left edge of the block in a small uppercase monospace style. Code will retain horizontal scrolling, readable line height, and mobile-safe sizing.

No copy button, line numbering, or code toolbar is added in this change.

## Files and compatibility

- `docs/style.css` becomes light-only and loses theme-toggle styling.
- `docs/script.js` keeps navigation and header behavior only.
- `docs/note.css` owns inline-code layout and code-block framing.
- `docs/note.js` owns language normalization, labels, and highlighting.
- `docs/index.html`, `docs/note.html`, and `docs/llm/*.html` lose theme controls.
- `docs/vendor/catppuccin-mocha.css` contains the local Highlight.js theme.
- Existing note URLs and the legacy compatibility HTML URLs remain valid.

Static resource version parameters will be incremented so browsers do not combine the old theme scripts or styles with the new HTML.

## Testing and acceptance

Automated tests will verify:

- no theme-toggle markup or theme persistence logic remains;
- the global stylesheet exposes only the approved light palette;
- every Markdown code fence has a language identifier;
- all viewer pages load the full browser Highlight.js bundle and local Catppuccin stylesheet;
- language normalization and visible-label behavior are covered by JavaScript tests;
- existing note topology, path safety, and viewer compatibility tests still pass.

Browser acceptance will cover the homepage and a representative Python, CUDA, Bash, and Text block. It will verify syntax spans have non-default Catppuccin colors, language labels are visible, inline code uses the pink treatment, no theme toggle remains, there are no console errors, and desktop plus 390-pixel layouts have no horizontal page overflow.
