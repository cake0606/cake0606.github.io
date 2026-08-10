# Homepage Simplification and Notes Reorganization Design

## Objective

Simplify the personal website into a focused, single-column homepage containing only Projects, Notes, and Plan. Reorganize infrastructure and LLM Markdown notes into a consistent two-level directory hierarchy, expose that hierarchy through an accessible expandable note tree, and replace the light theme's blue-green accents with a restrained Morandi pink palette on a pure white background.

## Scope

The update covers the homepage, shared homepage styling and behavior, the Markdown note viewer, note paths, automated structural checks, and browser verification.

The following content remains outside the homepage navigation and is not reorganized in this change:

- `docs/code-agent/`
- `docs/elementwise.md`
- `docs/learn.md`
- Existing standalone note `.html` files under `docs/llm/`

The standalone note `.html` files remain as compatibility pages, but the redesigned homepage does not link to them.

## Homepage Information Architecture

The header retains the site identity, theme control, and three navigation links:

1. Projects
2. Notes
3. Plan

The Hero, About, Contact, Current Focus, project cards, and explanatory footer content are removed. The main content uses a narrow single-column reading width and presents the three sections vertically. Sections are separated with whitespace and thin divider lines rather than parallel cards or multi-column grids.

### Projects

Projects remains visible as a section, but contains no existing projects. It shows only a concise empty state until projects are intentionally published later.

### Notes

Notes contains two collapsed first-level directories: Infra and LLM. Each first-level directory expands to reveal second-level directories. Each second-level directory expands to reveal the Markdown files it contains.

The homepage hierarchy is:

```text
Infra
├─ cuda
├─ nano-vllm
└─ vllm

LLM
├─ rl
│  ├─ ppo.md
│  └─ grpo.md
└─ concepts
   ├─ concepts.md
   ├─ gae.md
   ├─ index.md
   └─ kvcache.md
```

### Plan

Plan remains visible as a section with no plan entries. It shows a concise empty state. Future entries will use a square completion indicator followed by the plan text; completed entries may use a filled or checked square, while incomplete entries use an empty square.

## Visual Design

The light theme uses a pure white page background and warm charcoal text. Morandi pink is applied selectively so the interface remains quiet and readable.

| Role | Color | Usage |
| --- | --- | --- |
| Background | `#FFFFFF` | Page and primary section background |
| Primary text | `#3F3638` | Headings and body text |
| Muted text | `#746A6C` | Empty states and secondary labels |
| Deep pink | `#8E5F68` | Active navigation, links, directory titles |
| Mid pink | `#B98991` | Focus rings, icons, future plan squares |
| Soft pink | `#F4E8EA` | Expanded directory and hover surfaces |
| Mist pink | `#FAF5F6` | Nested directory background when needed |
| Divider | `#E8DDDF` | Section and row separators |

Large pink background blocks are avoided. Pink provides hierarchy through navigation state, directory state, links, focus treatment, and future Plan completion controls. The dark theme remains available and retains sufficient contrast, with only targeted changes needed to keep shared components consistent.

## Directory Reorganization

The Markdown files move to the following physical layout:

```text
docs/
├─ infra/
│  ├─ cuda/
│  │  ├─ cuda内存.md
│  │  ├─ elementwise.md
│  │  ├─ 点乘_softmax_norm.md
│  │  └─ 基础.md
│  ├─ nano-vllm/
│  │  ├─ kvcache_and_paged_attention.md
│  │  ├─ llm_engine.md
│  │  ├─ ModelRunner.md
│  │  ├─ run_nano-vllm.md
│  │  ├─ Scheduler.md
│  │  └─ struct.md
│  └─ vllm/
│     ├─ sampling.md
│     └─ scheduler.md
└─ llm/
   ├─ rl/
   │  ├─ grpo.md
   │  └─ ppo.md
   └─ concepts/
      ├─ concepts.md
      ├─ gae.md
      ├─ index.md
      └─ kvcache.md
```

The old `docs/cuda/`, `docs/nano-vllm/`, and `docs/vllm/` directories no longer exist after the move. Markdown files moved out of the root of `docs/llm/` no longer remain duplicated at their old paths.

## Interaction Design

The homepage directory tree uses native semantic disclosure controls. Infra and LLM are collapsed on initial load. Selecting a first-level directory toggles its second-level directories. Selecting a second-level directory toggles the Markdown file list. Multiple first- or second-level directories may remain open simultaneously.

Each Markdown filename is a normal link to the shared note viewer in the current tab. Links use the form:

```text
note.html?path=infra/vllm/scheduler.md
note.html?path=llm/rl/ppo.md
```

The note viewer validates relative Markdown paths, loads the selected file, displays the active path, and builds its note navigation from the active second-level directory. Sibling notes in that directory are visible in the left navigation, and the active note is highlighted. Invalid or missing paths show an explicit error instead of silently opening an unrelated default note.

## Accessibility and Responsive Behavior

- Directory toggles expose their expanded state through native disclosure semantics.
- Markdown entries remain real links and support keyboard navigation and standard browser behaviors.
- Focus indicators use the mid pink accent and remain visible on white and dark surfaces.
- Section headings preserve a logical document outline.
- The page remains a single column on desktop and mobile; spacing contracts at smaller widths without introducing horizontal scrolling.
- Reduced-motion preferences continue to disable nonessential animation.

## Verification Strategy

Automated tests use repository-local, dependency-free checks to verify:

- The homepage navigation and main content contain Projects, Notes, and Plan, with removed sections absent.
- Projects and Plan contain no published entries.
- The Infra and LLM directory hierarchy is represented on the homepage.
- Every homepage Markdown link resolves to an existing file below `docs/`.
- New Markdown locations exist and obsolete Infra directory locations do not.
- The light theme defines a pure white background and the approved Morandi pink tokens.
- The note viewer accepts the new relative paths and does not fall back silently for invalid paths.

Browser verification covers:

- First- and second-level directory expansion.
- Simultaneously open directory groups.
- Navigation from a Markdown filename to the correct note.
- Active sibling navigation in the note viewer.
- Light theme colors and readable contrast.
- Single-column layout at desktop and mobile widths.

## Acceptance Criteria

1. The homepage visibly contains only Projects, Notes, and Plan as content modules.
2. No previous project is displayed.
3. Plan has no entries but retains a clear future checkbox-based structure.
4. Infra and LLM expand through two directory levels before exposing Markdown links.
5. Every exposed Markdown link opens the corresponding note successfully.
6. The on-disk note hierarchy matches the approved Infra and LLM structure.
7. The light theme has a pure white background with the approved Morandi pink accents and no blue-green primary accents.
8. The design is single-column and usable with keyboard, desktop, and mobile navigation.
