const planGroups = document.getElementById("plan-groups");
const planState = document.getElementById("plan-state");
const header = document.querySelector(".site-header");
const navLinks = Array.from(document.querySelectorAll(".nav-link"));
const sections = navLinks
  .map((link) => document.querySelector(link.getAttribute("href")))
  .filter(Boolean);

function validatePlanData(data) {
  if (!Array.isArray(data)) {
    throw new TypeError("Plan data must be an array.");
  }

  data.forEach((group, groupIndex) => {
    if (!group || typeof group !== "object" || Array.isArray(group)) {
      throw new TypeError(`Plan group ${groupIndex} must be an object.`);
    }
    if (typeof group.title !== "string" || !group.title.trim()) {
      throw new TypeError(`Plan group ${groupIndex} must have a non-empty title.`);
    }
    if (!Array.isArray(group.items)) {
      throw new TypeError(`Plan group ${groupIndex} must have an items array.`);
    }

    group.items.forEach((item, itemIndex) => {
      if (!item || typeof item !== "object" || Array.isArray(item)) {
        throw new TypeError(`Plan item ${groupIndex}:${itemIndex} must be an object.`);
      }
      if (typeof item.text !== "string" || !item.text.trim()) {
        throw new TypeError(`Plan item ${groupIndex}:${itemIndex} must have non-empty text.`);
      }
      if (typeof item.completed !== "boolean") {
        throw new TypeError(`Plan item ${groupIndex}:${itemIndex} must have a boolean completed state.`);
      }
    });
  });

  return data;
}

function createPlanGroup(group) {
  const section = document.createElement("section");
  section.className = "plan-group";

  const title = document.createElement("h3");
  title.className = "plan-group-title";
  title.textContent = group.title;

  const list = document.createElement("ul");
  list.className = "plan-list";
  list.setAttribute("aria-label", `${group.title} plan`);

  group.items.forEach((item) => {
    const listItem = document.createElement("li");
    listItem.className = "plan-item";
    listItem.classList.toggle("is-complete", item.completed);

    const status = document.createElement("span");
    status.className = "plan-status";
    status.setAttribute("role", "img");
    status.setAttribute("aria-label", item.completed ? "已完成" : "未完成");

    const text = document.createElement("span");
    text.className = "plan-text";
    text.textContent = item.text;

    listItem.append(status, text);
    list.append(listItem);
  });

  section.append(title, list);
  return section;
}

function showPlanState(message) {
  planGroups.replaceChildren();
  planGroups.setAttribute("aria-busy", "false");
  planState.textContent = message;
  planState.hidden = false;
}

function renderPlanData(data) {
  const groups = data.map(createPlanGroup);
  planGroups.replaceChildren(...groups);
  planGroups.setAttribute("aria-busy", "false");

  if (groups.length === 0) {
    planState.textContent = "No plans published yet.";
    planState.hidden = false;
    return;
  }

  planState.hidden = true;
}

async function loadPlan() {
  if (!planGroups || !planState) return;

  try {
    const response = await fetch("plan.json", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Plan request failed with HTTP ${response.status}.`);
    }
    renderPlanData(validatePlanData(await response.json()));
  } catch (error) {
    console.error("Failed to load plan data:", error);
    showPlanState("Plans are temporarily unavailable.");
  }
}

loadPlan();

function updateHeaderState() {
  header?.classList.toggle("is-scrolled", window.scrollY > 8);
}

function setActiveLink(id) {
  navLinks.forEach((link) => {
    link.classList.toggle("is-active", link.getAttribute("href") === `#${id}`);
  });
}

if (sections.length > 0) {
  const observer = new IntersectionObserver(
    (entries) => {
      const visibleEntry = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

      if (visibleEntry) {
        setActiveLink(visibleEntry.target.id);
      }
    },
    {
      rootMargin: "-28% 0px -55% 0px",
      threshold: [0.05, 0.25, 0.5]
    }
  );

  sections.forEach((section) => observer.observe(section));
}

navLinks.forEach((link) => {
  link.addEventListener("click", () => {
    setActiveLink(link.getAttribute("href").slice(1));
  });
});

window.addEventListener("scroll", updateHeaderState, { passive: true });

updateHeaderState();
setActiveLink(window.location.hash.slice(1) || "projects");
